"""Volume 유무에 따른 감정요소(기쁨/슬픔/분노/중립) 단독 추가 효과 비교.
seasonal_volume_composition_single_comparison.csv 기준 — 기쁨의 개선 효과가
Volume 통제 하에서만 나타난다는 걸 표+그래프로 정리."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DELIVERABLE = Path(__file__).resolve().parents[1] / "data" / "suicide_predict" / "deliverable"
df = pd.read_csv(DELIVERABLE / "seasonal_volume_composition_single_comparison.csv")
s = df.set_index("model")

GROUPS = ["기쁨", "슬픔", "분노", "중립"]
base_vol = s.loc["Seasonal + Volume (기준선)", "MAE"]
base_novol = s.loc["Seasonal only (기준선, volume 없이)", "MAE"]

rows = []
for g in GROUPS:
    mae_vol = s.loc[f"Seasonal + Volume + E_{g}_mean", "MAE"]
    mae_novol = s.loc[f"Seasonal + E_{g}_mean (volume 없이)", "MAE"]
    rows.append(dict(
        감정=g,
        MAE_volume있음=mae_vol, dMAE_volume있음=mae_vol - base_vol,
        MAE_volume없음=mae_novol, dMAE_volume없음=mae_novol - base_novol,
    ))
table = pd.DataFrame(rows)
table.loc[len(table)] = ["기준선(Seasonal±Volume)", base_vol, 0.0, base_novol, 0.0]
print(table.to_string(index=False))
table.to_csv(DELIVERABLE / "seasonal_composition_volume_interaction_table.csv", index=False)

# ── 그래프: 감정별 ΔMAE, Volume 있음/없음 그룹 막대 ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(GROUPS))
w = 0.35

d_vol = [s.loc[f"Seasonal + Volume + E_{g}_mean", "MAE"] - base_vol for g in GROUPS]
d_novol = [s.loc[f"Seasonal + E_{g}_mean (volume 없이)", "MAE"] - base_novol for g in GROUPS]

b1 = ax.bar(x - w/2, d_vol, w, label="Volume 포함 (기준 MAE=%.1f)" % base_vol,
            color=["#2a78d6" if v < 0 else "#8fb8e8" for v in d_vol])
b2 = ax.bar(x + w/2, d_novol, w, label="Volume 없음 (기준 MAE=%.1f)" % base_novol,
            color="#e34948", alpha=0.85)

all_vals = d_vol + d_novol
ax.set_ylim(min(all_vals) - 3.2, max(all_vals) + 2.2)

for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + (0.35 if h >= 0 else -0.35),
                f"{h:+.1f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

ax.axhline(0, color="#333333", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(GROUPS)
ax.tick_params(axis="x", pad=8)
ax.set_ylabel("기준선 대비 ΔMAE (음수 = 개선)")
ax.set_title("감정요소 단독 추가 효과 — Volume 유무 비교")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()

out_png = DELIVERABLE / "seasonal_composition_volume_interaction_plot.png"
fig.savefig(out_png, dpi=150)
print(f"\nsaved: {DELIVERABLE / 'seasonal_composition_volume_interaction_table.csv'}")
print(f"saved: {out_png}")
