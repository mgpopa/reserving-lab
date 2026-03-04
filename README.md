# Reserving Lab (Chain Ladder + Uncertainty + Bootstrap)

This project is a small actuarial reserving lab built to explore:
- claims development triangles (cumulative and incremental)
- deterministic Chain Ladder ultimates and IBNR
- uncertainty (Mack-style standard errors and CVs)
- bootstrap reserve distributions and tail risk metrics (VaR / TVaR)
- simple stress scenarios (tail factor and inflation shocks)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py