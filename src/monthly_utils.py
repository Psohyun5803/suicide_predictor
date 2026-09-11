"""Monthly suicide count prediction utilities (MLP ensemble)."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "config"))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from huggingface_hub import hf_hub_download

from suicide_config import (
    EMO, RAW, EMO_GROUPS, GROUP_ORDER,
    REPO, BASE_CSV, K_TOPIC,
)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_socio():
    p  = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=BASE_CSV)
    df = pd.read_csv(p)
    df["month"] = pd.PeriodIndex(df["date"], freq="M")
    df = df.rename(columns={"자살자수": "y"})
    socio = [c for c in df.columns if c not in ("date", "month", "y")]
    return df[["month", "y"] + socio], socio


def monthly_emotion():
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError(
            "\n[ERROR] 전처리 파일이 없습니다. comment_kote_probs.parquet 필요"
        )
    df = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                         columns=["date"] + [f"emotion_{i}" for i in range(44)])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    labels  = json.load(open(EMO / "emotion_labels.json"))
    msum    = df.groupby("month")[[f"emotion_{i}" for i in range(44)]].sum()
    feats   = {}
    for g in GROUP_ORDER:
        members = [labels.index(e) for e in EMO_GROUPS[g] if e in labels]
        feats[f"E_{g}"] = np.log1p(msum[[f"emotion_{i}" for i in members]].sum(axis=1))
    return pd.DataFrame(feats).reset_index()


def monthly_topic():
    df = pd.read_parquet(RAW / "youtube_news_comments.parquet", columns=["date", "title"])
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    df["title"] = df["title"].fillna("").astype(str)
    uniq = df.drop_duplicates("title")["title"]
    uniq = uniq[uniq.str.len() > 0]
    vec  = TfidfVectorizer(max_features=20000, min_df=3, token_pattern=r"(?u)\b\w\w+\b")
    X    = vec.fit_transform(uniq)
    km   = KMeans(n_clusters=K_TOPIC, n_init=5, random_state=42).fit(X)
    t2c  = dict(zip(uniq, km.predict(X)))
    df["topic"] = df["title"].map(t2c)
    df = df.dropna(subset=["topic"]); df["topic"] = df["topic"].astype(int)
    ct  = (df.groupby(["month", "topic"]).size()
             .unstack(fill_value=0)
             .reindex(columns=range(K_TOPIC), fill_value=0))
    ct.columns = [f"T_{k}" for k in range(K_TOPIC)]
    return np.log1p(ct).reset_index()


def build_monthly():
    socio, S = load_socio()
    E = monthly_emotion(); T = monthly_topic()
    df = (socio.merge(E, on="month", how="inner")
               .merge(T, on="month", how="inner")
               .sort_values("month").reset_index(drop=True))
    Ecols = [c for c in df.columns if c.startswith("E_")]
    Tcols = [c for c in df.columns if c.startswith("T_")]
    return df, S, Ecols, Tcols


_SOC_KEYWORDS = [
    # 실업률 관련
    '실업', '실업자', '실업자수', '실직', '청년실업률',
    # 경제활동인구 관련
    '구직활동', '경제활동', '고용시장', '노동시장',
    # 비경제활동인구 관련
    '잠재구직자', '잠재취업가능자', '구직포기', '취업준비생', '구직단념자', '쉬었음',
    # 고용률 관련
    '취업', '취업자수', '취업자', '청년고용률',
    # 소비자물가 관련
    '물가', '소비자물가',
    # 가계신용 관련
    '가계빚', '가계부채', '가계대출', '판매신용', '주담대', '신용대출',
    # GDP 관련
    '국내총생산', '경제성장률',
    # 임금총액 관련
    '임금', '월급', '연봉', '봉급', '급여', '급여총액',
    # 근로 관련
    '노동시간', '소정근로일수', '휴일근로일수',
    # 뉴스 헤드라인 경제 용어
    '경제', '금리', '기준금리', '주가', '증시', '코스피', '코스닥',
    '환율', '수출', '무역', '부동산', '전세', '주택',
    '경기침체', '인플레',
    # 고용 형태·해고
    '비정규직', '정규직', '해고', '정리해고', '권고사직', '희망퇴직',
    '최저임금', '시급',
    # 자영업·소상공인
    '자영업', '소상공인', '폐업', '창업', '프리랜서',
    # 복지·실업급여
    '실업급여', '고용보험', '기초생활', '수급자', '차상위', '국민연금',
    # 경기·소비·파산
    '불황', '소비', '소비심리', '파산', '도산', '구조조정', '내수',
]


def monthly_emotion_by_category():
    """댓글 감정을 사경/비사경 뉴스별로 분리해 월별 집계.

    comment_id로 youtube_news_comments(제목)와
    comment_kote_probs(감정)를 조인해서 뉴스 카테고리별로 집계.
    """
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError("comment_kote_probs.parquet 필요")

    pat = "|".join(_SOC_KEYWORDS)

    # 제목 → is_soc 매핑
    news = pd.read_parquet(RAW / "youtube_news_comments.parquet",
                           columns=["comment_id", "title"])
    news["is_soc"] = news["title"].fillna("").str.contains(pat, na=False)
    id2soc = dict(zip(news["comment_id"], news["is_soc"]))
    del news

    # 감정 + 날짜
    emo = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                          columns=["comment_id", "date"] +
                          [f"emotion_{i}" for i in range(44)])
    emo["month"] = pd.PeriodIndex(pd.to_datetime(emo["date"]), freq="M")
    emo["is_soc"] = emo["comment_id"].map(id2soc).fillna(False)

    labels = json.load(open(EMO / "emotion_labels.json"))
    ecols = [f"emotion_{i}" for i in range(44)]

    frames = []
    for suffix, mask in [("soc", emo["is_soc"]), ("non", ~emo["is_soc"])]:
        msum = emo[mask].groupby("month")[ecols].sum()
        feats = {}
        for g in GROUP_ORDER:
            members = [labels.index(e) for e in EMO_GROUPS[g] if e in labels]
            feats[f"E_{g}_{suffix}"] = np.log1p(
                msum[[f"emotion_{i}" for i in members]].sum(axis=1)
            )
        frames.append(pd.DataFrame(feats).reset_index())

    return frames[0].merge(frames[1], on="month", how="outer").fillna(0)


def build_monthly_cat():
    """E_soc, E_non, E_all, S를 포함한 카테고리 분리 데이터 반환."""
    socio, S = load_socio()
    E_all = monthly_emotion()
    E_cat = monthly_emotion_by_category()

    df = (socio.merge(E_all, on="month", how="inner")
               .merge(E_cat, on="month", how="inner")
               .sort_values("month").reset_index(drop=True))

    Ecols_all = [c for c in df.columns
                 if c.startswith("E_") and not c.endswith(("_soc", "_non"))]
    Ecols_soc = [c for c in df.columns if c.endswith("_soc")]
    Ecols_non = [c for c in df.columns if c.endswith("_non")]

    return df, S, Ecols_all, Ecols_soc, Ecols_non


def _daily_group_stats(sub_emo, labels, suffix):
    """일별 감정 → 월별 (mean, std, slope) 통계.

    sub_emo: is_soc 필터링된 DataFrame, 'month'·'date_dt'·emotion_* 컬럼 필요
    returns: DataFrame indexed by month, 4그룹 × 3통계 = 12 컬럼
    """
    ecols_all = [f"emotion_{i}" for i in range(44)]
    daily = sub_emo.groupby(["month", "date_dt"])[ecols_all].sum()

    records = {}
    for g in GROUP_ORDER:
        members = [labels.index(e) for e in EMO_GROUPS[g] if e in labels]
        g_daily = np.log1p(daily[[f"emotion_{i}" for i in members]].sum(axis=1))
        g_daily.name = "val"
        g_df = g_daily.reset_index()

        for month, grp in g_df.groupby("month"):
            vals = grp["val"].values
            n = len(vals)
            mean_v = float(vals.mean())
            std_v  = float(vals.std()) if n > 1 else 0.0
            if n > 1:
                x = np.arange(n, dtype=float) - (n - 1) / 2
                slope = float(np.dot(x, vals) / (np.dot(x, x) + 1e-8))
            else:
                slope = 0.0
            rec = records.setdefault(month, {})
            rec[f"E_{g}_{suffix}_mean"]  = mean_v
            rec[f"E_{g}_{suffix}_std"]   = std_v
            rec[f"E_{g}_{suffix}_slope"] = slope

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "month"
    return df.reset_index()


def monthly_emotion_daily_stats_by_category():
    """일별 감정 집계 → 월별 (mean, std, slope), soc/non 분리.

    기존 monthly_emotion_by_category()의 단순 합산 대신
    일별 시계열 통계를 월 피처로 사용.
    반환 컬럼: E_{그룹}_{soc|non}_{mean|std|slope}  총 24개
    """
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError("comment_kote_probs.parquet 필요")

    pat = "|".join(_SOC_KEYWORDS)
    ecols = [f"emotion_{i}" for i in range(44)]

    news = pd.read_parquet(RAW / "youtube_news_comments.parquet",
                           columns=["comment_id", "title"])
    news["is_soc"] = news["title"].fillna("").str.contains(pat, na=False)
    id2soc = dict(zip(news["comment_id"], news["is_soc"]))
    del news

    emo = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                          columns=["comment_id", "date"] + ecols)
    emo["date_dt"] = pd.to_datetime(emo["date"]).dt.normalize()
    emo["month"]   = pd.PeriodIndex(emo["date_dt"], freq="M")
    emo["is_soc"]  = emo["comment_id"].map(id2soc).fillna(False)

    labels = json.load(open(EMO / "emotion_labels.json"))

    df_soc = _daily_group_stats(emo[emo["is_soc"]],  labels, "soc")
    df_non = _daily_group_stats(emo[~emo["is_soc"]], labels, "non")
    return df_soc.merge(df_non, on="month", how="outer").fillna(0)


def monthly_emotion_daily_stats():
    """일별 감정 집계 → 월별 (mean, std, slope), 전체(soc+non 합산).

    기존 monthly_emotion()의 단순 합산 대신 일별 시계열 통계.
    반환 컬럼: E_{그룹}_{mean|std|slope}  총 12개
    """
    if not (EMO / "comment_kote_probs.parquet").exists():
        raise FileNotFoundError("comment_kote_probs.parquet 필요")

    ecols = [f"emotion_{i}" for i in range(44)]
    emo = pd.read_parquet(EMO / "comment_kote_probs.parquet",
                          columns=["comment_id", "date"] + ecols)
    emo["date_dt"] = pd.to_datetime(emo["date"]).dt.normalize()
    emo["month"]   = pd.PeriodIndex(emo["date_dt"], freq="M")

    labels = json.load(open(EMO / "emotion_labels.json"))
    df_all = _daily_group_stats(emo, labels, "all")

    # 컬럼명을 E_{그룹}_{stat} 형태로 정리 (suffix 'all' 제거)
    df_all.columns = [
        c.replace("_all_", "_") if "_all_" in c else c
        for c in df_all.columns
    ]
    return df_all


def build_monthly_cat_daily():
    """일별 감정 통계(mean/std/slope) 기반 카테고리 분리 데이터 반환.

    E_soc: 4그룹 × 3통계 = 12피처
    E_non: 4그룹 × 3통계 = 12피처
    E_all: 4그룹 × 3통계 = 12피처
    """
    socio, S = load_socio()
    E_all = monthly_emotion_daily_stats()
    E_cat = monthly_emotion_daily_stats_by_category()

    df = (socio.merge(E_all, on="month", how="inner")
               .merge(E_cat, on="month", how="inner")
               .sort_values("month").reset_index(drop=True))

    Ecols_all = [c for c in df.columns if c.startswith("E_") and "_soc_" not in c and "_non_" not in c]
    Ecols_soc = [c for c in df.columns if "_soc_" in c]
    Ecols_non = [c for c in df.columns if "_non_" in c]

    return df, S, Ecols_all, Ecols_soc, Ecols_non


def build_monthly_cat_mean_var():
    """월별 감정 합산(mean) + 일별 분산(std) 8차원 피처.

    E_soc / E_non / E_all 각각:
      - E_{그룹}_{soc|non}       : 기존 log1p 월별 합산  (4피처)
      - E_{그룹}_{soc|non}_var   : 일별 값의 std          (4피처)
      합계 8피처
    """
    socio, S = load_socio()

    # 기존 월별 합산 (4차원)
    E_sum = monthly_emotion()
    E_cat_sum = monthly_emotion_by_category()

    # 일별 통계에서 std만 추출
    E_daily = monthly_emotion_daily_stats()
    E_cat_daily = monthly_emotion_daily_stats_by_category()

    std_all = E_daily[[c for c in E_daily.columns if c == "month" or c.endswith("_std")]]
    std_all.columns = ["month"] + [c.replace("_std", "_var") for c in std_all.columns if c != "month"]

    std_cat = E_cat_daily[[c for c in E_cat_daily.columns if c == "month" or c.endswith("_std")]]
    std_cat.columns = ["month"] + [c.replace("_std", "_var") for c in std_cat.columns if c != "month"]

    df = (socio
          .merge(E_sum,     on="month", how="inner")
          .merge(std_all,   on="month", how="inner")
          .merge(E_cat_sum, on="month", how="inner")
          .merge(std_cat,   on="month", how="inner")
          .sort_values("month").reset_index(drop=True))

    # E_all: 월별 합산 4 + 분산 4 = 8
    Ecols_all = (
        [c for c in df.columns if c.startswith("E_") and "_soc" not in c and "_non" not in c and not c.endswith("_var")]
        + [c for c in df.columns if c.startswith("E_") and "_soc" not in c and "_non" not in c and c.endswith("_var")]
    )
    # E_soc: 합산 4 + 분산 4 = 8
    Ecols_soc = (
        [c for c in df.columns if "_soc" in c and not c.endswith("_var")]
        + [c for c in df.columns if "_soc" in c and c.endswith("_var")]
    )
    # E_non: 합산 4 + 분산 4 = 8
    Ecols_non = (
        [c for c in df.columns if "_non" in c and not c.endswith("_var")]
        + [c for c in df.columns if "_non" in c and c.endswith("_var")]
    )

    return df, S, Ecols_all, Ecols_soc, Ecols_non


def build_monthly_infer(target_month):
    target_period = pd.Period(target_month, freq="M")
    train_df, S, Ecols, Tcols = build_monthly()
    train_df = train_df[train_df["y"].notna()].copy()

    socio_raw, _ = load_socio()
    s_row = socio_raw[socio_raw["month"] == target_period]
    has_S = len(s_row) > 0 and not s_row[S].isnull().any().any()

    E_df = monthly_emotion()
    e_row = E_df[E_df["month"] == target_period]
    _Ecols = [c for c in E_df.columns if c != "month"]
    has_E = len(e_row) > 0

    T_df = monthly_topic()
    t_row = T_df[T_df["month"] == target_period]
    _Tcols = [c for c in T_df.columns if c != "month"]
    has_T = len(t_row) > 0

    target_feats = {}
    if has_S:
        for c in S:
            target_feats[c] = float(s_row[c].values[0])
    if has_E:
        for c in _Ecols:
            target_feats[c] = float(e_row[c].values[0])
    if has_T:
        for c in _Tcols:
            target_feats[c] = float(t_row[c].values[0])

    return train_df, target_feats, S, Ecols, Tcols, has_S, has_E, has_T


# ── MLP ensemble ─────────────────────────────────────────────────────────────

_MLP_SEEDS = [0, 1, 2, 3]


class _MLP(nn.Module):
    def __init__(self, d_in, hidden=16, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp(Xtr, ytr, d_in, seed, epochs=400, wd=1e-2, dropout=0.4, val_k=4):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1.0
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8

    k = min(val_k, max(1, len(Xtr) // 6))
    Xt_raw, Xv_raw = Xtr[:-k], Xtr[-k:]
    yt_raw, yv_raw = ytr[:-k], ytr[-k:]

    Xt = torch.tensor((Xt_raw - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((yt_raw - ymu) / ysd, dtype=torch.float32)
    Xv = torch.tensor((Xv_raw - mu) / sd, dtype=torch.float32)
    yv = torch.tensor((yv_raw - ymu) / ysd, dtype=torch.float32)
    model = _MLP(d_in, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=wd)
    lossf = nn.SmoothL1Loss()
    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        lossf(model(Xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), yv).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k_: v.clone() for k_, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 60:
                break
    model.load_state_dict(best_state)
    return model, (mu, sd, ymu, ysd)


def _predict_mlp(model, stats, x):
    mu, sd, ymu, ysd = stats
    xn = torch.tensor((x - mu) / sd, dtype=torch.float32)
    with torch.no_grad():
        return model(xn).numpy() * ysd + ymu


def monthly_infer_predict(train_df, target_feats, cols):
    """train_df[cols] 전체 학습 → target_feats 단일 예측 (MLP, 4-seed 앙상블)."""
    X_train = train_df[cols].values.astype(np.float64)
    y_train = train_df["y"].values.astype(np.float64)
    X_pred  = np.array([[target_feats[c] for c in cols]])
    seed_preds = [_predict_mlp(*_train_mlp(X_train, y_train, len(cols), s), X_pred)[0]
                  for s in _MLP_SEEDS]
    return float(np.mean(seed_preds))


def expanding_eval(df, cols, start="2022-01", min_train=18):
    X = df[cols].values.astype(np.float64); y = df["y"].values.astype(np.float64)
    m = df["month"].astype(str).values
    ys, ps, ms = [], [], []
    for i in range(len(df)):
        if i < min_train or m[i] < start:
            continue
        seed_preds = [_predict_mlp(*_train_mlp(X[:i], y[:i], len(cols), s), X[i:i+1])[0]
                      for s in _MLP_SEEDS]
        ps.append(float(np.mean(seed_preds))); ys.append(y[i]); ms.append(m[i])
    ys, ps = np.array(ys), np.array(ps)
    r2 = 1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    metrics = dict(R2=r2, MAE=np.abs(ys - ps).mean(),
                   RMSE=np.sqrt(((ys - ps) ** 2).mean()), n=len(ys))
    preds = pd.DataFrame({"month": ms, "y_true": ys.round(1), "y_pred": ps.round(1)})
    return metrics, preds


def baseline_eval(df, kind, start="2022-01", min_train=18):
    y = df["y"].values; mo = df["month"].dt.month.values
    m = df["month"].astype(str).values
    ys, ps = [], []
    for i in range(len(df)):
        if i < min_train or m[i] < start:
            continue
        if kind == "seasonal_mean":
            same = y[:i][mo[:i] == mo[i]]
            ps.append(same.mean() if len(same) else y[:i].mean())
        elif kind == "lag12":
            ps.append(y[i - 12] if i >= 12 else y[:i].mean())
        ys.append(y[i])
    ys, ps = np.array(ys), np.array(ps)
    return dict(R2=1 - ((ys - ps) ** 2).sum() / ((ys - ys.mean()) ** 2).sum(),
                MAE=np.abs(ys - ps).mean(),
                RMSE=np.sqrt(((ys - ps) ** 2).mean()), n=len(ys))
