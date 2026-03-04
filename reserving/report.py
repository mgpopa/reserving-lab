import numpy as np

def reserve_summary_template(ay, ult, ibnr, se_ibnr=None, cv_ibnr=None, total_dist=None) -> str:
    total_ibnr = float(np.nansum(ibnr))
    top_ay = int(np.nanargmax(ibnr))
    lines = []
    lines.append("Reserve summary (draft)")
    lines.append("")
    lines.append(f"- Total IBNR (best estimate): {total_ibnr:,.0f}")
    lines.append(f"- Largest IBNR by accident year: AY {ay[top_ay]} ({ibnr[top_ay]:,.0f})")

    if se_ibnr is not None and cv_ibnr is not None:
        worst_cv = int(np.nanargmax(cv_ibnr))
        lines.append(f"- Highest relative uncertainty (CV): AY {ay[worst_cv]} (CV {cv_ibnr[worst_cv]:.2f})")

    if total_dist is not None and len(total_dist) > 10:
        p50 = np.percentile(total_dist, 50)
        p90 = np.percentile(total_dist, 90)
        p99 = np.percentile(total_dist, 99)
        tvar99 = total_dist[total_dist >= p99].mean()
        lines.append(f"- Risk view (bootstrap): P50 {p50:,.0f}, P90 {p90:,.0f}, P99 {p99:,.0f}, TVaR(99) {tvar99:,.0f}")

    lines.append("")
    lines.append(" To include:")
    lines.append("- What triangle represents (paid or incurred), and any data quirks.")
    lines.append("- Key assumptions: stable development pattern, tail factor choice, inflation handling.")
    lines.append("- Limitations: recent AYs have higher uncertainty, structural changes can break patterns.")
    return "\n".join(lines)