"""E(감정) = log1p(월별 확률 합계)는 댓글량(volume)과 감정강도(composition)가
섞여있음. 이를 분리:

  - volume       : log1p(월별 댓글 수)
  - E_*_mean     : 댓글당 평균 그룹 확률 (합계를 댓글 수로 나눔, volume-무관)

이후 계절성+volume을 통제한 뒤에도 emotion composition 자체가 추가
예측력을 갖는지 확인 (기존과 동일 MLP hidden=16/dropout=0.4/4seed 조건).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import numpy as np
import pandas as pd
from suicide_config import EMO, GROUP_ORDER, EMO_GROUPS
from monthly_utils import build_monthly, load_socio, expanding_eval

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"


def monthly_volume_and_composition():
    """comment_kote_probs.parquet -> (volume, E_*_mean) 월별 산출.

    volume    : log1p(월별 댓글 수)
    E_*_mean  : 댓글당 평균 그룹 확률 = 그룹 확률 합계 / 댓글 수  (기존 log1p(합계)와 달리
                댓글 수가 늘어도 커지지 않음 — 순수 "평균 감정 강도")
    """
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError("comment_kote_probs.parquet 필요")

    df = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                         columns=["date"] + [f"emotion_{i}" for i in range(44)])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    labels = json.load(open(EMO / "emotion_labels.json"))

    n_comments = df.groupby("month").size().rename("n_comments")
    msum = df.groupby("month")[[f"emotion_{i}" for i in range(44)]].sum()

    feats = {"volume": np.log1p(n_comments)}
    for g in GROUP_ORDER:
        members = [labels.index(e) for e in EMO_GROUPS[g] if e in labels]
        group_sum = msum[[f"emotion_{i}" for i in members]].sum(axis=1)
        feats[f"E_{g}_mean"] = group_sum / n_comments  # 댓글당 평균 (volume-무관)

    out = pd.DataFrame(feats).reset_index()
    out["n_comments"] = n_comments.values
    return out


def build_monthly_vc():
    """socio(S) + volume + E_*_mean 결합 데이터."""
    socio, S = load_socio()
    vc = monthly_volume_and_composition()
    df = socio.merge(vc, on="month", how="inner").sort_values("month").reset_index(drop=True)
    Ecomp_cols = [f"E_{g}_mean" for g in GROUP_ORDER]
    return df, S, Ecomp_cols


def run():
    df, S, Ecomp = build_monthly_vc()
    cal_month = df["month"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * cal_month / 12)
    df["season_cos"] = np.cos(2 * np.pi * cal_month / 12)
    SEASON = ["season_sin", "season_cos"]
    VOLUME = ["volume"]

    print(f"[data] {len(df)}개월 {df.month.min()}..{df.month.max()} | "
          f"volume 평균 {df.n_comments.mean():.0f}건/월, E_composition={len(Ecomp)}차원\n")
    print("월별 댓글 수 요약:")
    print(df[["month", "n_comments"]].to_string(index=False))

    combos = {
        "Seasonal only":                       SEASON,
        "Seasonal + Volume":                   SEASON + VOLUME,
        "Seasonal + Volume + E_composition":   SEASON + VOLUME + Ecomp,
        "Seasonal + E_composition (volume 없이)": SEASON + Ecomp,
    }
    rows = []
    for name, cols in combos.items():
        m, _ = expanding_eval(df, cols, start="2022-01")
        rows.append(dict(model=name, n_feat=len(cols),
                         R2=round(m["R2"], 4), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
        print(f"\n  {name:38s} n_feat={len(cols):3d}  R2={m['R2']:+.4f}  "
              f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}")

    res = pd.DataFrame(rows)
    s = res.set_index("model")
    print("\n=== 계절성+volume 통제 후에도 emotion composition이 추가 예측력을 갖는가 ===")
    d = s.loc["Seasonal + Volume + E_composition", "MAE"] - s.loc["Seasonal + Volume", "MAE"]
    print(f"  Seasonal+Volume -> +E_composition   ΔMAE {d:+.2f}  "
          f"(R² {s.loc['Seasonal + Volume','R2']:+.4f} -> "
          f"{s.loc['Seasonal + Volume + E_composition','R2']:+.4f})")
    print("  (음수 ΔMAE = composition이 volume과 별개로 추가 개선)")

    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    out_csv = DELIVERABLE / "seasonal_volume_composition_comparison.csv"
    res.to_csv(out_csv, index=False)

    vc_out = DELIVERABLE / "monthly_volume_composition.csv"
    monthly_volume_and_composition().to_csv(vc_out, index=False)
    print(f"\nsaved: {out_csv}")
    print(f"saved: {vc_out}  (월별 volume + E_*_mean 원본)")


if __name__ == "__main__":
    run()
