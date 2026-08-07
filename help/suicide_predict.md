# suicide_predict — 월별 자살 사망자수 예측 모델

## 개요

뉴스 댓글 기반 감정 지표와 사회경제 변수를 결합하여 월별 자살 사망자수를 예측합니다.

- **입력**: 사회경제 변수(S, 24개) + KOTE 감정(E, 4개) + 뉴스 토픽(T, 10개)
- **출력**: 해당 월 자살 사망자수
- **모델**: MLP 4-seed 앙상블 × 7가지 피처 조합
- **평가**: 확장 윈도우(expanding window) 1-step-ahead

---

## 파일 구조

```
suicide_predict/
├── config/suicide_config.py       # 상수·경로·설정
├── src/monthly_utils.py           # 데이터 로딩·모델·평가 함수
├── pipeline/monthly_pipeline.py   # 실행 파이프라인
├── help/                          # 문서
└── data/suicide_predict/
    ├── cache/
    │   ├── emotion/               # KOTE 추론 결과
    │   ├── raw/                   # 유튜브 댓글 원본
    │   └── topic/                 # 토픽 피처
    └── deliverable/               # 산출물 (CSV, PNG)
```

---

## 실행

```bash
cd suicide_predict/pipeline
python monthly_pipeline.py
```

---

## 파이프라인 단계

### 데이터 로딩

| 단계 | 함수 | 입력 | 출력 |
|------|------|------|------|
| 사회경제 | `load_socio()` | HF `MindCastSogang/SuicideDataset` | month, y, 사회경제 24개 변수 |
| 감정 집계 | `monthly_emotion()` | `cache/emotion/comment_kote_probs.parquet` | 월별 E_기쁨/슬픔/분노/중립 (log1p) |
| 토픽 집계 | `monthly_topic()` | `cache/raw/youtube_news_comments.parquet` | 월별 T_0~T_9 (log1p) |

### 평가

- 7개 피처 조합(S / E / T / S+E / S+T / E+T / S+E+T) 각각 확장 윈도우 평가
- 기준선: 계절평균, lag-12

### 산출물

```
deliverable/
├── monthly_suicide_7models.csv     # 7조합 + 기준선 성능 (R², MAE, RMSE)
├── 14_monthly_suicide_7models.png  # R²·MAE 막대그래프 비교
└── perm_importance_SE.png          # S+E 모델 변수 중요도 (퍼뮤테이션)
```

---

## 모델 구조

### MLP (월별 예측기)

```
입력(d_in) → Linear(d_in→16) → ReLU → Dropout(0.4) → Linear(16→1)
```

- 4-seed 앙상블 (seed: 0, 1, 2, 3) — 예측값 평균
- 정규화: 학습 세트의 평균·표준편차로 (값 - 평균) / 표준편차
- 조기 종료: 학습 세트 끝 4개월을 검증으로 사용, patience=60 epoch
- 옵티마이저: AdamW(lr=5e-3, weight_decay=1e-2), SmoothL1Loss

### 확장 윈도우 평가

- 분석 기간: 2020.01 ~ 2023.10 (46개월)
- 최소 학습 기간: 18개월
- 테스트 구간: 2022.01 ~ 2023.10 (22개월)
- 매 시점마다 이전 전체 데이터로 학습 → 현재 시점 1개월 예측

---

## 성능 결과

**MLP 7조합 (test: 2022.01 ~ 2023.10, n=22개월)**

| 모델 | R² | MAE |
|---|---|---|
| **S+E** | **+0.348** | **64.8명** ← 최고 |
| S 단독 | +0.232 | 69.8명 |
| S+E+T | +0.299 | 71.1명 |
| 기준: 계절평균 | +0.390 | 69.9명 |

---

## 주의사항

- 댓글 데이터는 2020-03 ~ 2023-10만 존재 (2020-01~02 없음)
- `monthly-infer` 시 S(사회경제) 데이터는 정부 통계 지연으로 최신 월이 없을 수 있음
- 토픽 클러스터링(KMeans)은 전체 데이터로 fit
