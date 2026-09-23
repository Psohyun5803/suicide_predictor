"""학습에 쓰인 전체 월별 데이터를 하나의 CSV로 내보내기.
S(사회경제 24개) + E(감정 4개, 원본 log1p합) + T_0~T_9(토픽 KMeans 10클러스터) +
volume(댓글 수, log1p 및 원본) + y(자살자수) 전부 포함."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from monthly_utils import build_monthly
from seasonal_volume_comparison_pipeline import monthly_volume_and_composition

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"


def run():
    df, S, E, T = build_monthly()
    vol = monthly_volume_and_composition()[["month", "volume", "n_comments"]]
    df = df.merge(vol, on="month", how="inner").sort_values("month").reset_index(drop=True)

    cal_month = df["month"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * cal_month / 12)
    df["season_cos"] = np.cos(2 * np.pi * cal_month / 12)

    cols = ["month", "y", "season_sin", "season_cos"] + S + E + T + ["volume", "n_comments"]
    out = df[cols].copy()
    out["month"] = out["month"].astype(str)

    print(f"[data] {len(out)}개월 {out.month.min()}..{out.month.max()}")
    print(f"컬럼: month, y, season_sin/cos(2) + S({len(S)}) + E({len(E)}) + T({len(T)}, T_0~T_{len(T)-1}) + volume/n_comments(2)")
    print(f"총 컬럼 수: {len(out.columns)}")

    out_path = DELIVERABLE / "monthly_training_data_full.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"\n{out.head(3).to_string(index=False)}")


if __name__ == "__main__":
    run()
