import csv
import io
from pathlib import Path
from typing import Union

import pandas as pd

def load_triangle_csv(file: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    # read bytes/text from path or uploaded file
    if hasattr(file, "read"):
        raw = file.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    else:
        text = Path(file).read_text(encoding="utf-8", errors="replace")

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("CSV is empty")

    header = [h.strip() for h in rows[0]]
    n = len(header)

    fixed_rows = []
    for r in rows[1:]:
        r = [c.strip() for c in r]
        if len(r) < n:
            r = r + [""] * (n - len(r))
        elif len(r) > n:
            r = r[:n]
        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=header)

    # convert dev columns to numeric & blanks to NaN
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
    return df