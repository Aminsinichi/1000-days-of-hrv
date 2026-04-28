import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np
from datetime import timedelta
from scipy.stats import pearsonr

############################################################################################

def trend_plot_daily(
    combined,
    outcome,
    moving_average_window=7,
    normal_range_window=60,
    min_gap_days = 10, 
    show='both',
    lower_pct=20,               
    upper_pct=80,
    use_log=False
):

    trends = combined[['date', outcome]].copy()
    # detection series
    if use_log:
        det = np.log(trends[outcome].astype(float))
    else:
        det = trends[outcome].astype(float)

    SMA_det   = det.rolling(moving_average_window, min_periods=2).mean()
    lower_det = det.rolling(normal_range_window, min_periods=30).quantile(lower_pct/100.0)
    upper_det = det.rolling(normal_range_window, min_periods=30).quantile(upper_pct/100.0)

    trend_flag = np.zeros(len(trends), dtype=int)
    trend_flag[SMA_det > upper_det] =  1
    trend_flag[SMA_det < lower_det] = -1

    trend_flag_s = pd.Series(trend_flag, index=trends.index)
    is_change = trend_flag_s.ne(trend_flag_s.shift()) & (trend_flag_s != 0)

    flags = trends.loc[is_change, ['date']].copy()
    flags['trend'] = trend_flag_s.loc[is_change].values
    flags['direction'] = flags['trend'].map({1:'up', -1:'down'})

    flags = flags.sort_values('date').reset_index(drop=True)
    kept = []
    last = None
    for _, r in flags.iterrows():
        if last is None or (r['date'] - last).days >= min_gap_days:
            kept.append(r)
            last = r['date']
    flags = pd.DataFrame(kept) if kept else flags.iloc[0:0]

    if show != 'both':
        flags = flags[flags['direction'] == show].copy()

    if not flags.empty:
        flags = flags.sort_values('date').reset_index(drop=True)
        if show == 'both':
            flags['num'] = flags.groupby('direction').cumcount() + 1
        else:
            flags['num'] = np.arange(1, len(flags) + 1)

    # visualization series (from log to raw)
    raw_series = trends[outcome].astype(float)
    if use_log: # take the exponent 
        vis_series = raw_series
        SMA_vis   = np.exp(SMA_det)
        lower_vis = np.exp(lower_det)
        upper_vis = np.exp(upper_det)
    else:
        vis_series = raw_series
        SMA_vis   = SMA_det
        lower_vis = lower_det
        upper_vis = upper_det
    y_label = outcome.upper()

    #  plot
    dates = combined['date']
    plt.figure(figsize=(15, 6), dpi=300)
    plt.bar(dates, vis_series, color = "skyblue", label=outcome.upper())
    plt.plot(dates, SMA_vis, label='Trend Line')
    plt.fill_between(dates, lower_vis, upper_vis, alpha=0.2,
                     label=f'Percentile Range [{lower_pct:.0f}–{upper_pct:.0f}]')

    for _, row in flags.iterrows():
        x = row['date']
        idx_list = trends.index[trends['date'].eq(x)].tolist()
        if not idx_list:
            continue
        y = SMA_vis.loc[idx_list[0]]
        if pd.isna(y):
            continue
        marker = '^' if row['direction'] == 'up' else 'v'
        color = 'green' if row['direction'] == 'up' else 'red'
        plt.scatter([x], [y], marker=marker, s=90, color=color,
                    edgecolors='black', linewidths=0.6, zorder=5)

        if 'num' in row:
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            dy = (ymax - ymin) * 0.02  # 2% of range
            y_text = y + (dy if row['direction'] == 'up' else -dy)
            va = 'bottom' if row['direction'] == 'up' else 'top'
            plt.text(
                x, y_text, f"{int(row['num'])}",
                ha='center', va=va, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.6, alpha=0.85),
                zorder=6
            )


    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.xlabel('Date', fontweight="bold")
    plt.ylabel(y_label, fontweight="bold")
    plt.title(f'{outcome.upper()} over Time with Deviation Flags', fontweight="bold")
    plt.xticks(rotation=45)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.grid(axis="y", alpha = 0.5)
    plt.show()

    return flags

############################################################################################

def trend_note_match(events, flag_table, days_before = 10, days_after = 10):
    trend_matches = []
    eventn = 0
    for i, trend_row in flag_table.iterrows():
        trend_date = trend_row['date']
        if eventn != trend_date:
            eventn +=1 
        window_start = trend_date - timedelta(days= days_before) # how many days before
        window_end = trend_date + timedelta(days= days_after) # how many days after 

        nearby_events = events[(events['date'] >= window_start) & (events['date'] <= window_end)]
        for _, event_row in nearby_events.iterrows():
            trend_matches.append({
                'trend_number':eventn,
                'trend_date': trend_date,
                'trend_direction': trend_row['direction'],
                'log_date': event_row['date'],
                "log_category": event_row["log_category"],
                'log_specific': event_row['log_specific']
            })

    trend_matches = pd.DataFrame(trend_matches)
    trend_matches.sort_values(by='log_date', inplace=True)
    
    return trend_matches

############################################################################################

def plot_hrv_cv(df, value_col="rmssd", use_log=False, window_days=7, min_periods=5):

    s = df.set_index("date")[value_col].astype(float).sort_index().dropna()
    x = np.log(s) if use_log else s

    roll_mean = x.rolling(window_days, min_periods=min_periods).mean()
    roll_std  = x.rolling(window_days, min_periods=min_periods).std(ddof=0)
    cv = 100.0 * (roll_std / roll_mean)

    sma = cv.rolling(window_days, min_periods=1).mean()

    plt.figure(figsize=(12, 4), dpi=200)
    plt.plot(sma.index, sma, linewidth=2, color="black", label=f"{window_days}-post-wake-up MA CV")
    plt.title(f"Coefficient of Variation of {'ln' if use_log else ''}RMSSD", fontweight="bold")
    plt.ylabel("CV (%)", fontweight="bold")
    plt.xlabel("Date", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.tight_layout()
    plt.show()

############################################################################################

def hrv_onset(
    combined, 
    major_events, 
    days_before=14,
    days_after=14,
    outcome="rmssd",
    ma_window=5,
    sharey=True,
    figsize_per_event=(12, 3.2),  # width, height per subplot
    dpi=300,
    use_log=False
):
    if outcome not in combined.columns:
        raise ValueError(f"Column '{outcome}' not found in combined.")

    # base daily series
    df = combined[['date', outcome]].copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.dropna(subset=[outcome])

    # keep only events that have at least one point in window
    evt_rows = []
    for onset_str, label in major_events.items():
        onset = pd.to_datetime(onset_str).normalize()
        start = onset - pd.Timedelta(days=days_before)
        end   = onset + pd.Timedelta(days=days_after)
        win = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        if not win.empty:
            evt_rows.append((onset, label))

    if not evt_rows:
        print("No data available within the requested windows for any events.")
        return

    evt_rows.sort(key=lambda x: x[0])

    n = len(evt_rows)

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig_w = figsize_per_event[0] * ncols
    fig_h = max(figsize_per_event[1] * nrows, 2.5)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        sharex=True,
        sharey=sharey
    )

    axes = np.atleast_1d(axes).ravel()

    for k in range(n, nrows * ncols):
        axes[k].set_visible(False)


    rel_index = pd.Index(range(-days_before, days_after + 1), name='rel_day')

    ymins, ymaxs = [], []
    results_rows = []
    
    for ax, (onset, label) in zip(axes, evt_rows):
        start = onset - pd.Timedelta(days=days_before)
        end   = onset + pd.Timedelta(days=days_after)

        win = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        if win.empty:
            ax.set_visible(False)
            continue

        win['rel_day'] = (win['date'] - onset).dt.days
        series = (win.set_index('rel_day')[outcome]
                    .reindex(rel_index)
                    .astype(float))

        # centered moving average
        ma = series.rolling(ma_window, min_periods=1, center=True).mean()

        # baseline/post means on the MA, and % change vs baseline 
        pre_ma  = ma.loc[ma.index < 0].dropna()
        post_ma = ma.loc[ma.index >= 0].dropna()

        baseline_mean = pre_ma.mean() if len(pre_ma) else np.nan
        post_mean     = post_ma.mean() if len(post_ma) else np.nan

        if pd.notna(baseline_mean) and baseline_mean != 0 and pd.notna(post_mean):
            delta_pct = 100.0 * (post_mean - baseline_mean) / baseline_mean
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "n/a"

        pre_str  = f"{baseline_mean:.1f}" if pd.notna(baseline_mean) else "n/a"
        post_str = f"{post_mean:.1f}"     if pd.notna(post_mean)     else "n/a"

        non_na = series.dropna()
        ax.scatter(
            non_na.index.to_numpy(),
            non_na.to_numpy(),
            s=60,                    
            alpha=0.4,
            edgecolors='none',
            label='Daily value',
            zorder=2
        )

        ax.plot(
            rel_index, ma,
            linewidth=4.0,          
            label=f'Moving average (w={ma_window})',
            zorder=3
        )
        ax.axvline(0, linestyle='--', linewidth=2.5, color='black')  
        ax.grid(alpha=0.35, linewidth=0.8)                          
        ax.set_xlim(-days_before, days_after)
        title_date = onset.strftime('%Y-%m-%d')

        # include Δ% and the pre/post means in the title
        ax.set_title(
            f"{label}  —  onset: {title_date}  |  Δ% vs baseline: {delta_str}  "
            f"|  Pre-{outcome.upper()}: {pre_str}, Post-{outcome.upper()}: {post_str}",
            fontweight='bold',
            fontsize=13               
        )

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        ax.tick_params(axis='both', which='major', labelsize=15, width=1.5, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)


        finite_vals = np.concatenate([non_na.to_numpy(), ma.dropna().to_numpy()]) if len(non_na) else ma.dropna().to_numpy()
        if finite_vals.size:
            ymins.append(np.nanmin(finite_vals))
            ymaxs.append(np.nanmax(finite_vals))

        results_rows.append({
            "event_date": title_date,
            "label": label,
            "n_pre": int(series.loc[series.index < 0].dropna().size),
            "n_post": int(series.loc[series.index >= 0].dropna().size),
            "pre_mean": float(baseline_mean) if pd.notna(baseline_mean) else np.nan,
            "post_mean": float(post_mean) if pd.notna(post_mean) else np.nan,
            "delta_pct": float(delta_pct) if 'delta_pct' in locals() and isinstance(delta_pct, (int, float)) else np.nan,
        })

    if ymins and ymaxs:
        y_min, y_max = float(np.min(ymins)), float(np.max(ymaxs))
        pad = (y_max - y_min) * 0.08 if y_max > y_min else (y_max or 1.0) * 0.08
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(y_min - pad, y_max + pad)

    
    bottom_row_start = (nrows - 1) * ncols
    for i in range(bottom_row_start, bottom_row_start + ncols):
        if i < len(axes) and axes[i].get_visible():
            axes[i].set_xlabel("Relative day (0 = onset)", weight="bold", fontsize=13)  
    

    fig.suptitle("Post-wake-up " + outcome.upper() + " Event-Related Dynamics", y=0.995, fontsize=20, weight="bold")  
    span = days_before + days_after
    step = 1 if span <= 20 else 2 if span <= 40 else 5
    plt.xticks(range(-days_before, days_after + 1, step))
    for ax in axes:
        if ax.get_visible():
            ax.legend(loc='upper right', fontsize=11)  
            break

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    results_df = pd.DataFrame(results_rows, columns=[
        "event_date", "label", "n_pre", "n_post",
        "pre_mean", "post_mean", "delta_pct"
    ])
    return results_df

############################################################################################

def trend_plot_weekly(
    combined,
    outcome,
    normal_range_weeks=15,  
    min_gap_weeks=3,        
    show='down',            
    lower_pct=20,           
    upper_pct=80,            
    use_log=False            
):

    # weekly resample 
    weekly = combined.set_index("date").resample("W").first().reset_index()

    # detection scale (weekly value is the signal; I don't take a moving average)
    if use_log:
        det = np.log(weekly[outcome].astype(float))
    else:
        det = weekly[outcome].astype(float)

    q_low  = det.rolling(normal_range_weeks, min_periods=6).quantile(lower_pct/100.0)
    q_high = det.rolling(normal_range_weeks, min_periods=6).quantile(upper_pct/100.0)

    trend_flag = np.zeros(len(weekly), dtype=int)
    trend_flag[det > q_high] =  1
    trend_flag[det < q_low]  = -1

    trend_flag_s = pd.Series(trend_flag, index=weekly.index)
    is_change = trend_flag_s.ne(trend_flag_s.shift()) & (trend_flag_s != 0)

    flags = weekly.loc[is_change, ['date']].copy()
    flags['trend'] = trend_flag_s.loc[is_change].values
    flags['direction'] = flags['trend'].map({1:'up', -1:'down'})

    flags = flags.sort_values('date').reset_index(drop=True)
    kept, last = [], None
    for _, r in flags.iterrows():
        if last is None or ((r['date'] - last).days >= 7 * min_gap_weeks):
            kept.append(r)
            last = r['date']
    flags = pd.DataFrame(kept) if kept else flags.iloc[0:0]

    if show != 'both':
        flags = flags[flags['direction'] == show].copy()

    # numbering for flags 
    if not flags.empty:
        flags = flags.sort_values('date').reset_index(drop=True)
        if show == 'both':
            flags['num'] = flags.groupby('direction').cumcount() + 1
        else:
            flags['num'] = np.arange(1, len(flags) + 1)

    raw_series = weekly[outcome].astype(float)
    vis_series = raw_series
    if use_log:
        lower_vis, upper_vis = (np.exp(q_low), np.exp(q_high))
    else:
        lower_vis, upper_vis = (q_low, q_high)
    y_label = outcome.upper()
    trend_line = vis_series  # weekly value itself

    dates = weekly['date']
    plt.figure(figsize=(15, 6), dpi=300)
    plt.bar(dates, vis_series, label=outcome.upper(), color="skyblue")
    plt.plot(dates, trend_line, label='Weekly', linewidth=1.8)

    plt.fill_between(dates, lower_vis, upper_vis, alpha=0.2,
                     label=f'Percentile Range [{lower_pct:.0f}–{upper_pct:.0f}]')

    # helper: integer to Roman numerals
    def _to_roman(n: int) -> str:
        vals = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
        ]
        res = []
        for v, sym in vals:
            count, n = divmod(n, v)
            res.append(sym * count)
        return "".join(res)

    for _, row in flags.iterrows():
        x = row['date']
        idx_list = weekly.index[weekly['date'].eq(x)].tolist()
        if not idx_list:
            continue
        y = trend_line.loc[idx_list[0]]
        if pd.isna(y):
            continue
        marker = '^' if row['direction'] == 'up' else 'v'
        color = 'green' if row['direction'] == 'up' else 'red'
        plt.scatter([x], [y], marker=marker, s=90, color=color,
                    edgecolors='black', linewidths=0.6, zorder=5)

        if 'num' in row:
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            dy = (ymax - ymin) * 0.02  # 2% of range
            y_text = y + (dy if row['direction'] == 'up' else -dy)
            va = 'bottom' if row['direction'] == 'up' else 'top'
            plt.text(
                x, y_text, _to_roman(int(row['num'])),
                ha='center', va=va, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=0.6, alpha=0.85),
                zorder=6
            )

    plt.xlabel('Date', fontweight="bold")
    plt.ylabel(y_label, fontweight="bold")
    plt.title(f'Weekly {outcome.upper()} with Percentile Band & Trend Flags', fontweight="bold")
    plt.xticks(rotation=45)
    plt.legend()

    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.grid(axis="y", alpha = 0.5)
    plt.show()

    return flags

############################################################################################

def duration_summary(combined):
    
    x, y = "rmssd", "rmssd_1min"   # x = 2min, y = 1min

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "legend.frameon": False,
        "grid.alpha": 0.25
    })
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.50, wspace=0.25)

    # A: 7-day moving averages over time 
    ax_top = fig.add_subplot(gs[0, :])
    mutual = combined[["date", x, y]].dropna().sort_values("date")

    ma2 = mutual[x].rolling(7, min_periods=1).mean()
    ma1 = mutual[y].rolling(7, min_periods=1).mean()

    ax_top.plot(mutual["date"], ma2, label="RMSSD 2min 7-day MA", linewidth=2, color="black")
    ax_top.plot(mutual["date"], ma1, label="RMSSD 1min 7-day MA", linewidth=2, color = "blue")

    ax_top.set_title("A — 7-Day Moving Averages over Time")
    ax_top.set_xlabel("Date")
    ax_top.set_ylabel("RMSSD (ms)")
    ax_top.legend(fontsize=10)
    ax_top.tick_params(axis="x", rotation=45)
    ax_top.grid(True, axis="y")
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.2)

    # B: Correlation
    ax_corr = fig.add_subplot(gs[1, 0])
    sns.regplot(
        data=mutual, x=x, y=y, ax=ax_corr,
        scatter_kws={"alpha": 0.5, "s": 25},
        line_kws={"lw": 2, "color": "red"},
        color="black"
    )

    r, pval = pearsonr(mutual[x], mutual[y])
    p_text = "p < .001" if pval < .001 else f"p = {pval:.3f}".replace("0.", ".")

    ax_corr.text(0.05, 0.95, f"r = {r:.2f}; {p_text}",
                 transform=ax_corr.transAxes, fontsize=12, fontweight="bold",
                 va="top", ha="left")

    ax_corr.set_title("B — Regression: RMSSD 2min vs RMSSD 1min")
    ax_corr.set_xlabel("RMSSD 2min (ms)")
    ax_corr.set_ylabel("RMSSD 1min (ms)")
    ax_corr.grid(True)
    for spine in ax_corr.spines.values():
        spine.set_linewidth(1.2)

    # C: Bland–Altman (1min vs 2min)
    ax_ba = fig.add_subplot(gs[1, 1])
    a = mutual[y].to_numpy()   # 1min
    b = mutual[x].to_numpy()   # 2min
    m = (a + b) / 2.0
    d = a - b

    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd

    ax_ba.scatter(m, d, s=25, alpha=0.7, color="black")
    ax_ba.axhline(md, lw=3, color="red")
    ax_ba.axhline(loa_u, lw=1.5, linestyle="--", color="gray")
    ax_ba.axhline(loa_l, lw=1.5, linestyle="--", color="gray")

    ax_ba.set_title("C — Bland–Altman: RMSSD 1min vs RMSSD 2min")
    ax_ba.set_xlabel(f"Mean of {y.upper()} and {x.upper()} (ms)")
    ax_ba.set_ylabel(f"Difference ({y.upper()} - {x.upper()}) (ms)")

    # Annotate lines 
    ylims = ax_ba.get_ylim()
    xlims = ax_ba.get_xlim()
    xpad = xlims[0] + 0.02 * (xlims[1] - xlims[0])
    ax_ba.text(xpad, md,     f"Mean = {md:.2f}", va="bottom")
    ax_ba.text(xpad, loa_u,  f"+1.96 SD = {loa_u:.2f}", va="bottom")
    ax_ba.text(xpad, loa_l,  f"-1.96 SD = {loa_l:.2f}", va="top")
    ax_ba.set_ylim(ylims)
    ax_ba.grid(True)
    for spine in ax_ba.spines.values():
        spine.set_linewidth(1.2)

    for ax in fig.axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

    plt.tight_layout(pad=1.5)
    plt.show()

############################################################################################

def day_night(combined):

    x, y = "rmssd", "night_rmssd"

    fig, ax = plt.subplots(figsize=(15, 6))

    sns.histplot(
        combined[y].dropna(),
        label="Post-wake-up RMSSD",
        kde=True,
        stat="density",
        alpha=0.45,
        ax=ax,
        color="orange",
    )

    sns.histplot(
        combined[x].dropna(),
        label="Nocturnal RMSSD",
        kde=True,
        stat="density",
        alpha=0.45,
        ax=ax,
        color="blue",
    )

    ax.set_title("Distribution of Post-wake-up vs Nocturnal RMSSD")
    ax.set_xlabel("RMSSD")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

############################################################################################

def day_night_summary(combined, major_events, roll=7):

    x, y = "rmssd", "night_rmssd"

    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "legend.frameon": False,
        "grid.alpha": 0.25
    })

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    ax_top = fig.add_subplot(gs[0, :])

    # major events
    for date, label in major_events.items():
        event_date = pd.to_datetime(date)
        ax_top.axvline(
            event_date,
            ls="--",
            lw=2,
            color="gray",
            alpha=0.8,
        )
        ax_top.text(
            event_date,
            0.98,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="gray",
            weight = "bold",
            transform=ax_top.get_xaxis_transform(),
        )

    day_night = combined[["date", "rmssd", "night_rmssd"]].dropna()  # same subset as your function

    ax_top.plot(
        day_night["date"],
        day_night["rmssd"].rolling(roll, min_periods=1).mean(),
        label="Post-wake-up RMSSD",
        color="black"
    )
    ax_top.plot(
        day_night["date"],
        day_night["night_rmssd"].rolling(roll, min_periods=1).mean(),
        label=f'{"Nocturnal RMSSD"}',
        color="blue"
    )

    ax_top.set_title(f"A — {roll}-Day Moving Averages over Time")
    ax_top.legend()
    ax_top.set_xlabel("Date")
    ax_top.set_ylabel("RMSSD")

    # date formatting
    ax_top.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    for lbl in ax_top.get_xticklabels():
        lbl.set_rotation(45)

    ax_top.grid(True, axis="y")
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.2)

    ax_corr = fig.add_subplot(gs[1, 0])
    sns.regplot(data=combined, x=x, y=y, ax=ax_corr,
            scatter_kws={"alpha":0.7, "s":25, "color":"black"},
            line_kws={"lw":2, "color":"red"})

    ax_corr.set_title("B — Regression: Nocturnal vs Post-wake-up")
    ax_corr.set_xlabel(x.upper())
    ax_corr.set_ylabel(y.upper())
    # Use Pearson r and add p-value 
    mutual = combined[[x, y]].dropna()
    r, pval = pearsonr(mutual[x], mutual[y])
    p_text = "p < .001" if pval < .001 else f"p = {pval:.3f}".replace("0.", ".")
    ax_corr.text(0.02, 0.95, f"r = {r:.2f}; {p_text}", transform=ax_corr.transAxes, va="top", ha="left", fontweight = "bold")
    ax_corr.grid(True)
    for spine in ax_corr.spines.values():
        spine.set_linewidth(1.2)

    ax_ba = fig.add_subplot(gs[1, 1])
    day_night = combined[["date", y, x]].dropna()
    a = day_night[y].to_numpy()
    b = day_night[x].to_numpy()
    m = (a + b) / 2.0
    d = a - b
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd
    ax_ba.scatter(m, d, s=25, alpha=0.7, color="black")
    ax_ba.axhline(md, lw=3, color="red")
    ax_ba.axhline(loa_u, lw=1.5, linestyle="--", color="gray")
    ax_ba.axhline(loa_l, lw=1.5, linestyle="--", color="gray")
    ax_ba.set_title("C — Bland–Altman: Post-wake-up vs Nocturnal")
    ax_ba.set_xlabel(f"Mean of {y.upper()} and {x.upper()}")
    ax_ba.set_ylabel(f"Difference ({y.upper()} - {x.upper()})")
    ylims = ax_ba.get_ylim()
    xlims = ax_ba.get_xlim()
    ax_ba.text(xlims[0] + 0.02*(xlims[1]-xlims[0]), md, f"Mean = {md:.2f}", va="bottom")
    ax_ba.text(xlims[0] + 0.02*(xlims[1]-xlims[0]), loa_u, f"+1.96 SD = {loa_u:.2f}", va="bottom")
    ax_ba.text(xlims[0] + 0.02*(xlims[1]-xlims[0]), loa_l, f"-1.96 SD = {loa_l:.2f}", va="top")
    ax_ba.set_ylim(ylims)
    ax_ba.grid(True)
    for spine in ax_ba.spines.values():
        spine.set_linewidth(1.2)

    for ax in fig.axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

    plt.tight_layout(pad=1.5)
    plt.show()