import numpy as np

def volume_weighted_link_ratios(cum: np.ndarray) -> np.ndarray:
    # f_j = sum(C_{i,j+1}) / sum(C_{i,j}) over rows where both exists
    n_dev = cum.shape[1]
    f = np.full(n_dev - 1, np.nan)
    for j in range(n_dev - 1):
        num = 0.0
        den = 0.0
        for i in range(cum.shape[0]):
            a = cum[i, j]
            b = cum[i, j+1]
            if not np.isnan(a) and not np.isnan(b) and a > 0:
                num += b
                den += a
        f[j] = num / den if den > 0 else np.nan
    return f

def project_cumulative(cum: np.ndarray, link: np.ndarray, tail_factor: float = 1.0) -> np.ndarray:
    # fill missing cells using chain ladder factors
    proj = cum.copy()
    n_ay, n_dev = proj.shape
    for i in range(n_ay):
        for j in range(n_dev - 1):
            if np.isnan(proj[i, j+1]) and not np.isnan(proj[i, j]) and not np.isnan(link[j]):
                proj[i, j+1] = proj[i, j] * link[j]
    # apply tail factor to ultimate (last column)
    proj[:, -1] = proj[:, -1] * tail_factor
    return proj

def ultimates_and_ibnr(cum: np.ndarray, proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ultimate = projected last column
    ibnr = ultimate - latest observed cumulative
    """
    ult = proj[:, -1]
    latest = np.array([row[np.where(~np.isnan(row))[0][-1]] for row in cum], dtype=float)
    ibnr = ult - latest
    return ult, ibnr