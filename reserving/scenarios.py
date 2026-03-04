import numpy as np
from .chainladder import volume_weighted_link_ratios, project_cumulative, ultimates_and_ibnr

def apply_inflation_shock(cum: np.ndarray, shock_pct: float, last_k_diagonals: int = 1) -> np.ndarray:
    """
    inflate the latest observed cumulative values (proxy for sudden inflation)
    last_k_diagonals=1 inflates only the latest diagonal cell per AY
    """
    shocked = cum.copy()
    for i in range(shocked.shape[0]):
        idx = np.where(~np.isnan(shocked[i, :]))[0]
        if len(idx) == 0:
            continue
        # inflate last observed cell (simple, demo-friendly)
        j = idx[-1]
        shocked[i, j] = shocked[i, j] * (1 + shock_pct)
    return shocked

def run_chainladder(cum: np.ndarray, tail_factor: float) -> dict:
    link = volume_weighted_link_ratios(cum)
    proj = project_cumulative(cum, link, tail_factor=tail_factor)
    ult, ibnr = ultimates_and_ibnr(cum, proj)
    return {"link": link, "proj": proj, "ult": ult, "ibnr": ibnr}