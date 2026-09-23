"""리뷰어 요청: emotion composition(기쁨/슬픔/분노/중립)을 4개 묶음으로 넣지 말고
하나씩 따로 넣어서, 혹시 4개 중 특정 감정 하나만 진짜 신호가 있고 나머지가
노이즈라 묶어서 보면 묻히는 건 아닌지 확인.

Seasonal+Volume 기준선(Volume 포함)뿐 아니라 Seasonal only 기준선(Volume 없이)에서도
감정 1개씩만 추가한 조합을 비교 — seasonal_volume_composition_comparison.csv의
"Seasonal+Volume+E_composition"과 "Seasonal+E_composition(volume 없이)" 두 행 모두
4개 묶음 대신 단독으로 재검정."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from suicide_config import GROUP_ORDER
from monthly_utils import expanding_eval
from seasonal_volume_comparison_pipeline import build_monthly_vc

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"


def run():
    df, S, Ecomp = build_monthly_vc()
    cal_month = df["month"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * cal_month / 12)
    df["season_cos"] = np.cos(2 * np.pi * cal_month / 12)
    SEASON = ["season_sin", "season_cos"]
    VOLUME = ["volume"]

    print(f"[data] {len(df)}개월 {df.month.min()}..{df.month.max()}\n")

    combos = {
        "Seasonal only (기준선, volume 없이)": SEASON,
        "Seasonal + Volume (기준선)": SEASON + VOLUME,
    }
    for g in GROUP_ORDER:
        combos[f"Seasonal + Volume + E_{g}_mean"] = SEASON + VOLUME + [f"E_{g}_mean"]
    combos["Seasonal + Volume + E_composition(4개 전부)"] = SEASON + VOLUME + Ecomp
    for g in GROUP_ORDER:
        combos[f"Seasonal + E_{g}_mean (volume 없이)"] = SEASON + [f"E_{g}_mean"]
    combos["Seasonal + E_composition(4개 전부, volume 없이)"] = SEASON + Ecomp

    rows = []
    for name, cols in combos.items():
        m, _ = expanding_eval(df, cols, start="2022-01")
        rows.append(dict(model=name, n_feat=len(cols),
                         R2=round(m["R2"], 4), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
        print(f"  {name:42s} n_feat={len(cols):2d}  R2={m['R2']:+.4f}  "
              f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}")

    res = pd.DataFrame(rows)
    s = res.set_index("model")

    base_mae_vol = s.loc["Seasonal + Volume (기준선)", "MAE"]
    print("\n=== 기준선(Seasonal+Volume) 대비 ΔMAE (음수=개선) ===")
    for g in GROUP_ORDER:
        d = s.loc[f"Seasonal + Volume + E_{g}_mean", "MAE"] - base_mae_vol
        print(f"  +E_{g}_mean  ΔMAE {d:+.2f}")
    d_all = s.loc["Seasonal + Volume + E_composition(4개 전부)", "MAE"] - base_mae_vol
    print(f"  +전부(4개)   ΔMAE {d_all:+.2f}")

    base_mae_novol = s.loc["Seasonal only (기준선, volume 없이)", "MAE"]
    print("\n=== 기준선(Seasonal only, volume 없이) 대비 ΔMAE (음수=개선) ===")
    for g in GROUP_ORDER:
        d = s.loc[f"Seasonal + E_{g}_mean (volume 없이)", "MAE"] - base_mae_novol
        print(f"  +E_{g}_mean  ΔMAE {d:+.2f}")
    d_all2 = s.loc["Seasonal + E_composition(4개 전부, volume 없이)", "MAE"] - base_mae_novol
    print(f"  +전부(4개)   ΔMAE {d_all2:+.2f}")

    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    out_csv = DELIVERABLE / "seasonal_volume_composition_single_comparison.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    run()
