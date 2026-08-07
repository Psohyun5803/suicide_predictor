"""월별 자살 사망자수 예측 — 상수·경로·설정."""
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1] / "data" / "suicide_predict"
EMO         = ROOT / "cache" / "emotion"
TOP         = ROOT / "cache" / "topic"
RAW         = ROOT / "cache" / "raw"
DELIVERABLE = ROOT / "deliverable"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# ── 감정 분류 ────────────────────────────────────────────────────────────
GROUP_ORDER = ["기쁨", "슬픔", "분노", "중립"]

EMO_GROUPS = {
    "기쁨": ["환영/호의","감동/감탄","고마움","존경","기대감","뿌듯함","편안/쾌적",
             "신기함/관심","아껴주는","즐거움/신남","흐뭇함(귀여움/예쁨)","행복","기쁨","안심/신뢰"],
    "슬픔": ["슬픔","안타까움/실망","부끄러움","절망","패배/자기혐오","귀찮음","힘듦/지침",
             "죄책감","당황/난처","부담/안_내킴","서러움","재미없음","불쌍함/연민","불안/걱정"],
    "분노": ["불평/불만","지긋지긋","화남/분노","의심/불신","한심함","역겨움/징그러움",
             "짜증","어이없음","증오/혐오","경악"],
    "중립": ["우쭐댐/무시함","공포/무서움","비장함","없음","깨달음","놀람"],
}

# ── 데이터 소스 ──────────────────────────────────────────────────────────
REPO             = "MindCastSogang/SuicideDataset"
BASE_CSV         = "suicide_base_data_2020_2024_20260222.csv"
COMMENT_REPO     = "MindCastSogang/Youtube_news_preprocessed_data"
COMMENT_BASE_DIR = "preprocessed/v1"
KOTE_MODEL       = "searle-j/kote_for_easygoing_people"
KOTE_BATCH       = 512
KOTE_MAXLEN      = 128

# ── 모델 설정 ────────────────────────────────────────────────────────────
K_TOPIC = 10
