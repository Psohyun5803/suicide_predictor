"""Season / Volume / Emotion(Composition) 세 변수그룹의 3가지 조합
(Season+Emotion+Volume, Season+Emotion, Volume+Emotion) 각각에서
감정 4개(기쁨/슬픔/분노/중립)를 단독으로 추가했을 때의 효과를 표+그래프 3세트로 정리."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from suicide_config import GROUP_ORDER
from monthly_utils import expanding_eval, baseline_eval
from seasonal_volume_comparison_pipeline import build_monthly_vc

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable" / "triad"
DELIVERABLE.mkdir(parents=True, exist_ok=True)


def eval_group(df, base_cols, base_label):
    """base_label은 실제 투입 피처 그대로의 이름(예: 'Season+Volume') — 'Emotion'은
    아직 아무 감정도 안 넣은 상태를 가리키는 컨텍스트 이름일 뿐이라 행 이름엔 넣지 않음."""
    rows = []
    m0, _ = expanding_eval(df, base_cols, start="2022-01")
    rows.append(dict(model=f"기준선({base_label})", n_feat=len(base_cols),
                     R2=round(m0["R2"], 4), MAE=round(m0["MAE"], 3),
                     RMSE=round(m0["RMSE"], 3), n=m0["n"]))
    for g in GROUP_ORDER:
        m, _ = expanding_eval(df, base_cols + [f"E_{g}_mean"], start="2022-01")
        rows.append(dict(model=f"{base_label}+E_{g}_mean", n_feat=len(base_cols) + 1,
                         R2=round(m["R2"], 4), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
    return pd.DataFrame(rows)


def eval_emotion_alone(df, seasonal_ref_mae):
    """Season도 Volume도 없이 감정 하나만 입력으로 썼을 때 — 비교할 '기준선' 피처가
    없으므로, 참고용으로 계절평균(seasonal_mean) 나이브 기준선 MAE만 점선으로 표시."""
    rows = []
    for g in GROUP_ORDER:
        m, _ = expanding_eval(df, [f"E_{g}_mean"], start="2022-01")
        rows.append(dict(model=f"E_{g}_mean 단독", n_feat=1,
                         R2=round(m["R2"], 4), MAE=round(m["MAE"], 3),
                         RMSE=round(m["RMSE"], 3), n=m["n"]))
    rows.append(dict(model="참고: 계절평균(나이브)", n_feat=0,
                     R2=round(seasonal_ref_mae["R2"], 4), MAE=round(seasonal_ref_mae["MAE"], 3),
                     RMSE=round(seasonal_ref_mae["RMSE"], 3), n=seasonal_ref_mae["n"]))
    return pd.DataFrame(rows)


def plot_emotion_alone(table, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    body = table.iloc[:-1]
    ref_mae = table.iloc[-1]["MAE"]
    labels = GROUP_ORDER
    mae = body["MAE"].values
    colors = ["#2a78d6" if m < ref_mae else "#e34948" for m in mae]
    bars = ax.bar(labels, mae, color=colors, width=0.55)
    ax.axhline(ref_mae, color="#666666", linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, ref_mae + max(mae) * 0.02, f"참고: 계절평균 MAE={ref_mae:.1f}",
            ha="right", fontsize=9, color="#666666")
    for b, m in zip(bars, mae):
        ax.text(b.get_x() + b.get_width()/2, m + max(mae)*0.015, f"{m:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("MAE (낮을수록 좋음)")
    ax.set_title("Emotion 단독 (Season·Volume 없이) — 감정요소 개별 비교")
    ax.set_ylim(0, max(mae) * 1.18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_group(table, label, base_mae, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = ["기준선"] + GROUP_ORDER
    mae = table["MAE"].values
    colors = ["#999999"] + ["#2a78d6" if m < base_mae else "#e34948" for m in mae[1:]]
    bars = ax.bar(labels, mae, color=colors, width=0.6)
    ax.axhline(base_mae, color="#666666", linestyle="--", linewidth=1)
    for b, m in zip(bars, mae):
        ax.text(b.get_x() + b.get_width()/2, m + max(mae)*0.015, f"{m:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("MAE (낮을수록 좋음)")
    ax.set_title(f"{label} — 감정요소 단독 추가 비교")
    ax.set_ylim(0, max(mae) * 1.18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run():
    df, S, Ecomp = build_monthly_vc()
    cal_month = df["month"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * cal_month / 12)
    df["season_cos"] = np.cos(2 * np.pi * cal_month / 12)
    SEASON = ["season_sin", "season_cos"]
    VOLUME = ["volume"]

    # (컨텍스트 이름=파일명/제목용, 실제 피처 조합 이름=행 이름용)
    contexts = {
        "Season+Emotion+Volume": ("Season+Volume", SEASON + VOLUME),
        "Season+Emotion": ("Season", SEASON),
        "Volume+Emotion": ("Volume", VOLUME),
    }

    for ctx_label, (base_label, base_cols) in contexts.items():
        table = eval_group(df, base_cols, base_label)
        print(f"\n=== {ctx_label} ===")
        print(table.to_string(index=False))

        safe = ctx_label.lower().replace("+", "_")
        csv_path = DELIVERABLE / f"triad_{safe}.csv"
        table.to_csv(csv_path, index=False)

        png_path = DELIVERABLE / f"triad_{safe}_plot.png"
        plot_group(table, base_label, table["MAE"].iloc[0], png_path)
        print(f"saved: {csv_path}")
        print(f"saved: {png_path}")

    # 4번째: Emotion 단독 (Season도 Volume도 없이)
    seasonal_ref = baseline_eval(df, "seasonal_mean", start="2022-01")
    table4 = eval_emotion_alone(df, seasonal_ref)
    print("\n=== Emotion (단독, Season·Volume 없이) ===")
    print(table4.to_string(index=False))
    csv4 = DELIVERABLE / "triad_emotion_alone.csv"
    table4.to_csv(csv4, index=False)
    png4 = DELIVERABLE / "triad_emotion_alone_plot.png"
    plot_emotion_alone(table4, png4)
    print(f"saved: {csv4}")
    print(f"saved: {png4}")


if __name__ == "__main__":
    run()
