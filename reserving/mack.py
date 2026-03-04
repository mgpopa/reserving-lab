import numpy as np

def mack_sigmas(cum: np.ndarray, link: np.ndarray) -> np.ndarray:
    """
    estimate sigma_j^2 from observed link ratios:
      r_{i,j} = C_{i,j+1}/C_{i,j}
    """
    n_ay, n_dev = cum.shape
    sig2 = np.full(n_dev - 1, np.nan)

    for j in range(n_dev - 1):
        ratios = []
        weights = []
        for i in range(n_ay):
            a = cum[i, j]
            b = cum[i, j+1]
            if not np.isnan(a) and not np.isnan(b) and a > 0:
                ratios.append(b / a)
                weights.append(a)
        ratios = np.array(ratios, float)
        weights = np.array(weights, float)
        if len(ratios) >= 2 and not np.isnan(link[j]):
            # weighted sample variance around selected link
            mu = link[j]
            v = np.sum(weights * (ratios - mu) ** 2) / np.sum(weights)
            sig2[j] = v
    return sig2

def mack_se_ibnr(cum: np.ndarray, link: np.ndarray, tail_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    practical approximation:
    SE(ultimate_i) derived from factor variances beyond latest development
    """
    sig2 = mack_sigmas(cum, link)
    n_ay, n_dev = cum.shape

    latest_j = np.array([np.where(~np.isnan(cum[i, :]))[0][-1] for i in range(n_ay)], int)
    latest_c = np.array([cum[i, latest_j[i]] for i in range(n_ay)], float)

    ult = np.zeros(n_ay)
    se_ult = np.zeros(n_ay)

    for i in range(n_ay):
        j0 = latest_j[i]
        # project to last dev
        cur = latest_c[i]
        var = 0.0
        for j in range(j0, n_dev - 1):
            if np.isnan(link[j]) or np.isnan(sig2[j]):
                continue
            # multiplicative projection: variance approx accumulates proportional to cur^2 * sig2
            var += (cur ** 2) * sig2[j]
            cur = cur * link[j]
        cur = cur * tail_factor
        ult[i] = cur
        se_ult[i] = np.sqrt(max(var, 0.0))
    ibnr = ult - latest_c
    se_ibnr = se_ult  # approximate: uncertainty mainly in the future development
    cv_ibnr = np.where(ibnr != 0, se_ibnr / np.abs(ibnr), np.nan)
    return se_ibnr, cv_ibnr