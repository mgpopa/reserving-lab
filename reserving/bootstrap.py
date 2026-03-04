import numpy as np
from .triangle import cumulative_to_incremental, incremental_to_cumulative
from .chainladder import volume_weighted_link_ratios, project_cumulative, ultimates_and_ibnr

def bootstrap_total_ibnr(cum: np.ndarray, n_sims: int = 2000, seed: int = 42, tail_factor: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    inc = cumulative_to_incremental(cum)
    # collect residuals by development column
    residuals_by_dev = []
    for j in range(inc.shape[1]):
        col = inc[:, j]
        vals = col[~np.isnan(col)]
        # center to avoid drift
        vals = vals - np.mean(vals) if len(vals) else vals
        residuals_by_dev.append(vals)

    totals = []
    for _ in range(n_sims):
        inc_sim = inc.copy()
        # resample residuals for observed cells only
        for j in range(inc.shape[1]):
            vals = residuals_by_dev[j]
            if len(vals) == 0:
                continue
            for i in range(inc.shape[0]):
                if not np.isnan(inc_sim[i, j]):
                    inc_sim[i, j] = max(0.0, inc_sim[i, j] + rng.choice(vals))
        cum_sim = incremental_to_cumulative(inc_sim)

        link = volume_weighted_link_ratios(cum_sim)
        proj = project_cumulative(cum_sim, link, tail_factor=tail_factor)
        _, ibnr = ultimates_and_ibnr(cum_sim, proj)
        totals.append(np.nansum(ibnr))
    return np.array(totals, float)