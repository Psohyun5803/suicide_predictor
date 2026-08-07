"""월별 자살자수 예측 — 7모델 비교 파이프라인."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from monthly_utils import build_monthly, expanding_eval, baseline_eval

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"

def run_monthly():
    df, S, E, T = build_monthly()
    combos = {
        "1.사회경제(S)": S,
        "2.감정(E)": E,
        "3.토픽(T)": T,
        "4.S+E": S + E,
        "5.S+T": S + T,
        "6.E+T": E + T,
        "7.S+E+T": S + E + T,
    }
    rows = []
    for name, cols in combos.items():
        m, _ = expanding_eval(df, cols, start="2022-01")
        rows.append(dict(model=name, n_feat=len(cols),
                         R2=round(m["R2"], 3), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
        print(f"  {name:14s}  R²={m['R2']:+.3f}  MAE={m['MAE']:.1f}")

    for bname, bkind in [("기준:계절평균", "seasonal_mean"), ("기준:lag-12", "lag12")]:
        bm = baseline_eval(df, bkind, start="2022-01")
        rows.append(dict(model=bname, n_feat=0,
                         R2=round(bm["R2"], 3), MAE=round(bm["MAE"], 3),
                         RMSE=round(bm["RMSE"], 3), n=bm["n"]))

    out = DELIVERABLE / "monthly_suicide_7models.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved: {out}")

if __name__ == "__main__":
    run_monthly()
