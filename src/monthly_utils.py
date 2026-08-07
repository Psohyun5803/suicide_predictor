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
    Xn = (Xtr - mu) / sd; yn = (ytr - ymu) / ysd
    k = min(val_k, max(1, len(Xn) // 6))
    Xt = torch.tensor(Xn[:-k], dtype=torch.float32)
    yt = torch.tensor(yn[:-k], dtype=torch.float32)
    Xv = torch.tensor(Xn[-k:], dtype=torch.float32)
    yv = torch.tensor(yn[-k:], dtype=torch.float32)
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
