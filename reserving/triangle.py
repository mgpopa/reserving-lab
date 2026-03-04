import numpy as np
import pandas as pd

def to_matrix(df: pd.DataFrame) -> tuple[list[str], list[int], np.ndarray]:
    ay = df.iloc[:, 0].astype(str).tolist()
    dev = [int(c) for c in df.columns[1:]]
    mat = df.iloc[:, 1:].to_numpy(dtype=float)
    return ay, dev, mat

def cumulative_to_incremental(cum: np.ndarray) -> np.ndarray:
    inc = np.full_like(cum, np.nan)
    inc[:, 0] = cum[:, 0]
    for j in range(1, cum.shape[1]):
        inc[:, j] = cum[:, j] - cum[:, j-1]
    return inc

def incremental_to_cumulative(inc: np.ndarray) -> np.ndarray:
    cum = np.full_like(inc, np.nan)
    for i in range(inc.shape[0]):
        running = 0.0
        for j in range(inc.shape[1]):
            if np.isnan(inc[i, j]):
                cum[i, j] = np.nan
            else:
                running += inc[i, j]
                cum[i, j] = running
    return cum

def latest_diagonal(cum: np.ndarray) -> np.ndarray:
    # latest observed cumulative for each AY = last non-nan entry in row
    latest = []
    for i in range(cum.shape[0]):
        row = cum[i, :]
        idx = np.where(~np.isnan(row))[0]
        latest.append(row[idx[-1]] if len(idx) else np.nan)
    return np.array(latest, dtype=float)