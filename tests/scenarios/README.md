# 10-Speaker 테스트 시나리오 (A–G)

실제 회의 음성·화자 음성은 개인정보이므로 저장소에 포함하지 않습니다(`data/` 는 git-ignore).
아래 절차로 **사내에서** 데이터를 준비한 뒤 `scripts/evaluate.py` 로 측정합니다.

## 1. 데이터 준비

| 구분 | 위치 | 내용 |
|---|---|---|
| 등록용(enrollment) | 화자 등록 탭 또는 `meeting-transcriber enroll-dir <dir>` | 1인당 3개 이상, 각 10–30초 |
| 평가용 clip bank | `data/eval_clips/<이름>/*.wav` (+ 같은 이름 `.txt` 전사) | **등록에 쓰지 않은** 별도 발화 |
| 짧은 맞장구(F) | `data/eval_clips/<이름>/short/*.wav` | "네.", "맞습니다.", "잠시만요.", "아니요." 등 0.3–1.5초 |
| 실제 회의(권장) | 임의 경로 + 수작업 RTTM | 최종 판단은 실제 회의 녹음 기준 |

## 2. 시나리오 생성 (합성 회의, 정답 RTTM 자동 생성)

```bash
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset A --out data/eval/A
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset B --out data/eval/B
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset C --out data/eval/C
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset D --out data/eval/D --unenrolled 게스트1 게스트2
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset E --out data/eval/E
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset F --out data/eval/F
python scripts/make_synthetic_meeting.py --clips data/eval_clips --preset G --out data/eval/G --speakers <비슷한 음색 5명>
```

| ID | 목적 | 생성 파라미터 | 핵심 지표 |
|---|---|---|---|
| A | 2명 기본 | 2명, overlap 5% | DER, ID accuracy |
| B | 5명 | 5명 | DER, ID accuracy |
| C | 10명 (FIXED / RANGE 2–10 비교) | 10명 | DER, 화자 수 추정 오차, ID accuracy |
| D | 등록 8 + 미등록 2 | 10명, `--unenrolled` 2명 | **Unknown rejection, False accept** |
| E | 겹침 발화 | overlap 35% | DER(confusion), overlap 플래그 |
| F | 짧은 맞장구 다수 | interjection 60% | 짧은 발화 화자 오귀속, Miss |
| G | 비슷한 성별/음색 | 직접 선택 5명 | misidentification, margin 분포 |

`--snr 15` 등으로 잡음 조건을, `--seed` 로 반복 측정을 추가할 수 있습니다.

## 3. 측정

```bash
cp tests/scenarios/manifest.example.yaml data/eval/manifest.yaml   # 경로/이름 수정
python scripts/evaluate.py --manifest data/eval/manifest.yaml
python scripts/calibrate_threshold.py --scores outputs/eval_*/similarity_scores.json
```

리포트(`outputs/eval_*/report.md`)는 시나리오별로 `as_run / flexible / strict_one_to_one / argmax`
매칭 모드를 같은 유사도 행렬에서 재계산해 비교합니다. 합성 회의는 실제 회의보다 쉬운 조건(근접 마이크, 잔향 없음)
이므로 결과는 **상한(upper bound)** 으로 해석하고, 최종 threshold·기본 모드는 실제 회의 녹음으로 확정합니다.
