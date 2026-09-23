import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("/home/sohyun44/mindcast/suicide_predict/data/suicide_predict/deliverable/seasonal_volume_composition_single_comparison.csv")

labels = ["기준선", "+기쁨", "+슬픔", "+분노", "+중립"]
mae = df["MAE"].values[:5]
base = mae[0]
colors = ["#999999"] + ["#2a78d6" if m < base else "#e34948" for m in mae[1:]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, mae, color=colors, width=0.6)
ax.axhline(base, color="#666666", linestyle="--", linewidth=1)
ax.text(len(labels)-0.5, base+0.8, f"기준선 MAE={base:.1f}", ha="right", fontsize=9, color="#666666")

for b, m in zip(bars, mae):
    ax.text(b.get_x()+b.get_width()/2, m+0.8, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("MAE (낮을수록 좋음)")
ax.set_title("Seasonal+Volume 기준선 + 감정요소 개별 추가 비교")
ax.set_ylim(0, max(mae)*1.15)
fig.tight_layout()
fig.savefig("/home/sohyun44/mindcast/suicide_predict/data/suicide_predict/deliverable/seasonal_volume_composition_single_plot.png", dpi=150)
print("saved")
