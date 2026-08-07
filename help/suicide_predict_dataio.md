# suicide_predict — 데이터 입출력 명세 (월별 예측기)

## 전체 흐름

```
[HF: MindCastSogang/SuicideDataset]   [HF: Youtube_news_preprocessed_data]
              │                                       │
         load_socio()                         (사전 전처리 완료)
              │                                       │
   사회경제 변수 24개 (월별)              cache/emotion/comment_kote_probs.parquet
                                          cache/raw/youtube_news_comments.parquet
                                                      │
                                    ┌─────────────────┤
                              monthly_emotion()   monthly_topic()
                                    │                  │
                             E_기쁨/슬픔/분노/중립    T_0~T_9 (log1p)
                                    └────────┬─────────┘
                                             │
                                      build_monthly()
                                             │
                                     expanding_eval()
                                             │
                                       deliverable/
```

---

## 함수별 입출력

### load_socio()

| 구분 | 경로 | 형식 | 내용 |
|---|---|---|---|
| **입력** | HF `MindCastSogang/SuicideDataset` | csv (HF) | `suicide_base_data_2020_2024_20260222.csv` |
| **출력** | — (DataFrame) | — | `month`, `y`(자살자수), 사회경제 변수 24개 |

---

### monthly_emotion()

| 구분 | 경로 | 형식 | 내용 |
|---|---|---|---|
| **입력** | `cache/emotion/comment_kote_probs.parquet` | parquet | 댓글별 44개 감정 확률 (`emotion_0`~`emotion_43`) |
| **입력** | `cache/emotion/emotion_labels.json` | json | KOTE 44감정 레이블 순서 |
| **출력** | — (DataFrame) | — | `month`, `E_기쁨`, `E_슬픔`, `E_분노`, `E_중립` |

처리 방식: 감정 대분류별 월합계 → log1p 변환

---

### monthly_topic()

| 구분 | 경로 | 형식 | 내용 |
|---|---|---|---|
| **입력** | `cache/raw/youtube_news_comments.parquet` | parquet | `date`, `title` |
| **출력** | — (DataFrame) | — | `month`, `T_0`~`T_9` |

처리 방식: TF-IDF + KMeans(K=10) → 월별 토픽 댓글 수 → log1p 변환

---

### build_monthly()

| 구분 | 내용 |
|---|---|
| **입력** | `load_socio()` + `monthly_emotion()` + `monthly_topic()` inner join on `month` |
| **출력** | 46개월 × (2 + 24 + 4 + 10)차원 DataFrame, S/E/T 컬럼 리스트 |

---

### expanding_eval()

| 구분 | 내용 |
|---|---|
| **입력** | `build_monthly()` 결과 DataFrame, 사용할 컬럼 리스트 |
| **출력** | metrics dict (`R2`, `MAE`, `RMSE`, `n`), predictions DataFrame (`month`, `y_true`, `y_pred`) |

---

## 산출물

| 파일 | 형식 | 컬럼 |
|---|---|---|
| `deliverable/monthly_suicide_7models.csv` | csv | `model`, `n_feat`, `R2`, `MAE`, `RMSE`, `n` |
| `deliverable/14_monthly_suicide_7models.png` | png | R²·MAE 막대그래프 |
| `deliverable/perm_importance_SE.png` | png | S+E 모델 퍼뮤테이션 중요도 |

---

## 데이터 분할

| 구간 | 기간 | 용도 |
|---|---|---|
| 학습 누적 | 2020.01 ~ 직전 시점 | 확장 윈도우 학습 |
| 테스트 | 2022.01 ~ 2023.10 | 22개월 평가 |

---

## 환경변수

| 변수 | 설명 |
|---|---|
| `HF_TOKEN` | HuggingFace 인증 토큰 (`MindCastSogang` 조직 접근 권한 필요) |
