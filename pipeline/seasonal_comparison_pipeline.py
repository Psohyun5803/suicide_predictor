"""계절성(sin/cos)을 predictor로 직접 포함 — 계절성 통제 후에도 E/S가
추가 예측력을 갖는지 확인. 기존과 동일한 MLP(hidden=16, dropout=0.4,
4-seed[0,1,2,3]) + expanding-window 조건에서 4개 모델 비교.

  1. Seasonal only
  2. Seasonal + E
  3. Seasonal + S
  4. Seasonal + S + E
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from monthly_utils import build_monthly, expanding_eval

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"


def run_seasonal_comparison():
    df, S, E, T = build_monthly()
    cal_month = df["month"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * cal_month / 12)
    df["season_cos"] = np.cos(2 * np.pi * cal_month / 12)
    SEASON = ["season_sin", "season_cos"]

    print(f"[data] {len(df)}개월 {df.month.min()}..{df.month.max()} | S={len(S)} E={len(E)}\n")

    combos = {
        "Seasonal only":    SEASON,
        "Seasonal + E":     SEASON + E,
        "Seasonal + S":     SEASON + S,
        "Seasonal + S + E": SEASON + S + E,
    }
    rows = []
    for name, cols in combos.items():
        m, _ = expanding_eval(df, cols, start="2022-01")
        rows.append(dict(model=name, n_feat=len(cols),
                         R2=round(m["R2"], 4), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
        print(f"  {name:20s} n_feat={len(cols):3d}  R2={m['R2']:+.4f}  "
              f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  n={m['n']}")

    res = pd.DataFrame(rows)
    s = res.set_index("model")
    print("\n=== 계절성 통제 후에도 E/S가 추가 예측력을 갖는가 ===")
    dE  = s.loc["Seasonal + E",     "MAE"] - s.loc["Seasonal only", "MAE"]
    dS  = s.loc["Seasonal + S",     "MAE"] - s.loc["Seasonal only", "MAE"]
    dSE = s.loc["Seasonal + S + E", "MAE"] - s.loc["Seasonal + S",  "MAE"]
    print(f"  Seasonal      -> +E   ΔMAE {dE:+.2f}  "
          f"(R² {s.loc['Seasonal only','R2']:+.4f} -> {s.loc['Seasonal + E','R2']:+.4f})")
    print(f"  Seasonal      -> +S   ΔMAE {dS:+.2f}  "
          f"(R² {s.loc['Seasonal only','R2']:+.4f} -> {s.loc['Seasonal + S','R2']:+.4f})")
    print(f"  Seasonal + S  -> +E   ΔMAE {dSE:+.2f}  "
          f"(R² {s.loc['Seasonal + S','R2']:+.4f} -> {s.loc['Seasonal + S + E','R2']:+.4f})")
    print("  (음수 ΔMAE = 추가로 개선됨)")

    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    out = DELIVERABLE / "seasonal_mlp_comparison.csv"
    res.to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return res


if __name__ == "__main__":
    run_seasonal_comparison()
