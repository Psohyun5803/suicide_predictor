# 계절성 통제 분석 — Seasonal / Volume / Emotion Composition

## 개요

월별 자살 사망자수 예측 모델에서, (1) 계절성을 predictor로 직접 포함해도 감정(E)이
추가 예측력을 갖는지, (2) 기존 E(댓글량+감정강도가 혼재된 지표)를 댓글량(Volume)과
댓글당 평균 감정확률(Emotion Composition)로 분리했을 때도 감정 구성 자체가 추가
예측력을 갖는지 확인.

## 공통 방법론

| 항목 | 값 |
|---|---|
| 모델 | MLP: `Linear(d_in→16) → ReLU → Dropout(0.4) → Linear(16→1)` |
| 앙상블 | 4-seed (0, 1, 2, 3), 예측값 평균 |
| Optimizer | AdamW (lr=5e-3, weight_decay=1e-2) |
| Loss | SmoothL1Loss |
| 평가 | 확장 윈도우(expanding window), 2022-01 ~ 2023-10, n=22 |

**예측 방법론**: t-1월까지의 S, E로 MLP 모델을 학습한 후, t월의 S, E를 이용해
t월의 자살자수를 예측함(매 시점마다 그 이전 전체 데이터로 재학습).

**표준화 파라미터(train/valid/test)**: 표준화 파라미터(평균 및 표준편차)는
train/validation 분리 이전의 전체 corresponding training pool로부터 계산되며,
이렇게 산출된 동일한 파라미터가 training, validation, test(예측 대상월) 데이터
각각에 동일하게 적용됨. 

---

## 1. 계절성을 predictor로 직접 포함

**변수**: Seasonal = `sin(2π×month/12)`, `cos(2π×month/12)` (2개) / E = 기존
`monthly_emotion()` 방식(월별 확률합의 log1p, 4개) / S = 사회경제 변수(24개)

| 모델 | 변수 차원 | R² | MAE | RMSE |
|---|---|---|---|---|
| Seasonal only | 2 | 0.372 | 63.49 | 83.41 |
| **Seasonal + E** | 6 | **0.382** | **61.09** | 82.78 |
| Seasonal + S | 26 | 0.326 | 66.20 | 86.40 |
| Seasonal + S + E | 30 | 0.270 | 67.84 | 89.93 |

계절성을 predictor에 직접 추가한 이후에도 E는 추가 예측력을 확보함(R² 0.372→0.382,
MAE 63.49→61.09). 반면 S를 추가하면(Seasonal+S, Seasonal+S+E) 오히려 성능이
하락함.

관련 script => pipeline/seasonal_comparison_pipeline.py
---

## 2. 댓글량(Volume)과 감정구성(Emotion Composition)의 분리

**변수**: Seasonality(2) / Comment Volume = `log(1+n)`(월별 전체 댓글 수 n, 1개) /
Emotion Composition = 댓글당 평균 joy/sadness/anger/other 확률(그룹별 확률합 ÷
댓글수, log 변환 없음, 4개)

| 모델 | n_feat | R² | MAE | RMSE |
|---|---|---|---|---|
| Seasonality only | 2 | 0.372 | 63.49 | 83.41 |
| **Seasonality + Volume** | 3 | **0.476** | **51.72** | 76.18 |
| Seasonality + Volume + Emotion Composition | 7 | 0.160 | 67.85 | 96.46 |
| Seasonality + Emotion Composition | 6 | 0.011 | 77.66 | 104.70 |

Seasonality+Volume이 최고 성능이며, Emotion Composition이 추가되면 성능이
악화되고, Emotion Composition만 사용해도(Volume 없이) Seasonality only보다
성능이 좋지 않음을 확인함.

관련 script => pipeline/seasonal_volume_comparison_pipeline.py