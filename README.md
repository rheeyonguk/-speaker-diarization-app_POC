# 한국어 회의 다화자 전사 PoC (Local)

**결론:** 한국어 회의 음성/영상을 로컬에서 `WhisperX(전사) → WhisperX Alignment(단어 타임스탬프) → pyannote Community-1(화자 분리, 겹침 보존) → WeSpeaker(등록 화자 식별) → TXT/JSON/CSV/SRT` 로 처리하는 실행 가능한 PoC입니다. 등록되지 않았거나 확신이 부족한 화자는 이름을 붙이지 않고 `UNKNOWN_01, UNKNOWN_02 …` 로 구분합니다.

> ⚠ 이 시스템은 **100% 정확한 화자 분리를 보장하지 않습니다.** 결과는 자동 생성물이며 검토 후 사용해야 합니다(§12 정확도 영향 요인).
> ⚠ `speaker_match_threshold` 초기값(0.50)은 PoC 출발점일 뿐입니다. **환경별 calibration 필요** (§7.5).

---

## 목차
1. [아키텍처](#1-아키텍처)
2. [Upstream 조사 결과와 의존성 결정 근거](#2-upstream-조사-결과와-의존성-결정-근거)
3. [설치](#3-설치)
4. [필요한 사용자 작업 (Hugging Face 토큰 · Community-1 약관 승인 · 모델 사전 다운로드)](#4-필요한-사용자-작업)
5. [실행](#5-실행)
6. [결과 형식](#6-결과-형식)
7. [화자 등록 · 식별 설계](#7-화자-등록--식별-설계)
8. [Overlap 처리](#8-overlap-처리)
9. [평가 · 10-Speaker 테스트 시나리오](#9-평가--10-speaker-테스트-시나리오)
10. [설정](#10-설정)
11. [GPU / CPU / 장시간 음성](#11-gpu--cpu--장시간-음성)
12. [정확도에 영향을 미치는 요소와 한계](#12-정확도에-영향을-미치는-요소와-한계)
13. [보안 · 개인정보(음성정보)](#13-보안--개인정보음성정보)
14. [검증 현황](#14-검증-현황)
15. [트러블슈팅](#15-트러블슈팅)
16. [프로젝트 구조](#16-프로젝트-구조)

---

## 1. 아키텍처

```
입력(WAV/MP3/M4A/MP4/MOV)
 │  Layer 1-2  ffprobe 확인 → ffmpeg 16 kHz mono PCM 작업 사본(원본 불변) → peak 정규화(gain만) → [noise reduction hook, 기본 OFF]
 ▼
Layer 3  WhisperX ASR (faster-whisper / CTranslate2, VAD 배치 추론)   language=ko | auto
 ▼
Layer 4  WhisperX forced alignment (wav2vec2 CTC, 한국어 모델은 WhisperX 기본표에서 런타임 조회) → 단어 타임스탬프
 ▼
Layer 5  pyannote/speaker-diarization-community-1   AUTO | RANGE(min,max) | FIXED(num)
 │        DiarizeOutput.speaker_diarization          (겹침 보존 → overlap 플래그, 클러스터 통계)
 │        DiarizeOutput.exclusive_speaker_diarization (한 시점 1화자 → 단어 화자 할당)
 ▼
Layer 6-8  클러스터별 비겹침·긴 발화 구간 선택 → WeSpeaker 임베딩 → 클러스터 대표 임베딩
 │          × 등록 프로필(robust centroid) → cosine 유사도 행렬 → 매칭(flexible / strict_one_to_one) → 이름 | UNKNOWN_xx
 ▼
Layer 9  단어 단위 화자 할당 → 화자 전환 지점에서 발화 분할 → overlap / secondary_speakers 표시
 ▼
Export   TXT(회의록) · JSON(전체 메타데이터) · CSV · SRT
```

각 단계 종료 시 모델을 해제하고 `torch.cuda.empty_cache()` 를 **단계당 1회** 호출합니다(`runtime.release_models_between_stages=true`, 기본값). 최대 VRAM ≈ 가장 큰 단일 모델 수준으로 유지됩니다.

---

## 2. Upstream 조사 결과와 의존성 결정 근거

구현 전 각 저장소의 최신 소스·README·의존성을 직접 확인했습니다(조사 시점 2026-10-06). 설치된 실제 패키지에 대해 시그니처 회귀 테스트(`tests/unit/test_upstream_api.py`)를 둬서, 업그레이드로 API가 바뀌면 테스트가 실패하도록 했습니다.

| 구성요소 | 채택 버전 | 확인한 사실 (source of truth) | 결정 근거 |
|---|---|---|---|
| **Python** | 3.11 (지원 3.10–3.12) | whisperx `>=3.10,<3.14`, pyannote.audio `>=3.10`, WeSpeaker 분류자 최대 3.11 | 세 프로젝트 교집합 중 가장 검증된 버전 |
| **WhisperX** | `3.8.6` (PyPI 최신 정식) | main 은 `3.8.7rc1`(pre-release). `torch~=2.8.0`, `torchaudio~=2.8.0`, `torchcodec<0.8`, `pyannote-audio>=4.0.0`, `faster-whisper>=1.2.0`, `huggingface-hub<1.0` 고정. `DiarizationPipeline` 기본 모델 = **`pyannote/speaker-diarization-community-1`** 확인. 한국어 alignment 기본 모델 = `kresnik/wav2vec2-large-xlsr-korean` (`DEFAULT_ALIGN_MODELS_HF["ko"]`) | 정식 릴리스 사용, rc 회피. 정렬 모델명은 코드에 하드코딩하지 않고 런타임에 whisperx 표에서 읽음 |
| **pyannote.audio** | `4.0.7` (`>=4.0.7,<5`) | `Pipeline.from_pretrained(checkpoint, token=, cache_dir=)`; `apply(file, num_speakers, min_speakers, max_speakers, hook)` → **`DiarizeOutput(speaker_diarization, exclusive_speaker_diarization, speaker_embeddings)`** (legacy=True 일 때만 Annotation) | 3.x API 가정 안 함. 두 diarization 모두 보존 |
| **faster-whisper** | `1.2.1` | `large-v3`, `large-v3-turbo`(=`turbo`) 모델 맵 확인 | WhisperX 호환 최신 정식 |
| **CTranslate2** | `4.8.2` | Linux wheel 빌드 환경 = **CUDA 12.8 + cuDNN 9.10.2** | PyTorch 2.8.0 PyPI wheel(cu128, `nvidia-cudnn-cu12==9.10.2.21`)과 CUDA/cuDNN 메이저 일치 → 충돌 없음 |
| **PyTorch** | `2.8.0` (cu128) | WhisperX 가 `~=2.8.0` 로 고정 | 임의 업/다운그레이드 없이 WhisperX 제약을 그대로 따름 |
| **torchcodec** | `0.7.0` | WhisperX `<0.8`, pyannote `>=0.7.0` | 두 제약의 유일 교집합 |
| **WeSpeaker** | git commit `9fecd6c` | **PyPI 미배포**(공식 설치법이 `pip install git+…`). 태그 v1.2.0 은 Python `load_model`/허브 모델 지원 이전 버전. `wespeaker.load_model()` → `Speaker.extract_embedding_from_pcm()` API 확인 | 움직이는 branch 가 아니라 **불변 commit 해시로 고정**(재현성). 정식 릴리스가 나오면 교체 |
| **Gradio** | `6.17.3` | 최신 6.29.x 는 `huggingface-hub>=1.0` 계열을 요구 → WhisperX 3.8.6 의 `<1.0` 과 충돌하므로 해석기가 6.17.3 선택 | 다운그레이드가 아니라 상위 제약(WhisperX)에서 결정된 버전 |
| **nltk** | `3.10.3` (WhisperX 경유) | 3.10 부터 **프록시 경유 다운로드를 기본 거부**(SSRF 방어) → WhisperX alignment 가 첫 실행 시 `punkt_tab` 을 못 받으면 실패 | 사전 점검 + 명확한 오류 + `prefetch_models.py --allow-proxied-nltk` 제공 |

- 의존성 충돌 분석 순서(WhisperX → pyannote → WeSpeaker → PyTorch → CUDA → CTranslate2 → torchaudio)에 따라 확인한 결과 **충돌 없음**: `uv pip check` = "All installed packages are compatible"(166 패키지), 일반 `pip` 해석기 dry-run 도 동일 버전으로 해석.
- 검증된 정확한 버전 목록: `requirements-lock-linux-cu128.txt` (Linux x86_64 기준 constraints 파일).

---

## 3. 설치

### 3.1 사전 요구사항
| 항목 | 요구 |
|---|---|
| OS | Linux x86_64 또는 Windows 10/11 x64 |
| Python | 3.11 권장 (3.10–3.12) |
| ffmpeg | `ffmpeg`, `ffprobe` 가 PATH 에 있어야 함 (Windows: `winget install Gyan.FFmpeg`, Ubuntu: `sudo apt install ffmpeg`) |
| git | WeSpeaker 를 GitHub commit 에서 설치 |
| GPU(권장) | NVIDIA GPU + CUDA 12.x 를 지원하는 최신 드라이버 (`nvidia-smi` 의 CUDA Version ≥ 12.8 권장). VRAM 8 GB 이상 권장(large-v3 float16) |

### 3.2 Linux
```bash
git clone <this repo> && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"                     # PyPI torch 2.8.0 Linux wheel = CUDA 12.8 빌드 포함
# (선택) 검증된 버전 그대로:  pip install -c requirements-lock-linux-cu128.txt -e ".[dev]"
cp .env.example .env                         # HF_TOKEN 입력 (§4)
```
`uv` 사용 시: `uv venv -p 3.11 .venv && uv pip install -e ".[dev]"`

### 3.3 Windows (PowerShell)
PyPI 의 Windows용 torch 는 **CPU 전용**이므로, CUDA 빌드 torch 를 **먼저** 설치합니다(WhisperX 가 요구하는 2.8.0 계열).
```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
copy .env.example .env
```
- Windows 에서 `torchcodec` 은 FFmpeg **shared** 빌드 DLL 이 필요합니다. 이 앱은 디코딩을 ffmpeg CLI 로 하고 pyannote 에는 waveform 텐서를 직접 넘기므로 torchcodec 디코딩을 사용하지 않습니다. import 경고가 보여도 동작에는 영향이 없습니다.
- CTranslate2 GPU 추론에 cuBLAS/cuDNN 9 DLL 이 필요합니다. CUDA 빌드 torch 를 설치하면 함께 들어오며, 오류 시 §15 참고.

---

## 4. 필요한 사용자 작업

| # | 작업 | 이유 |
|---|---|---|
| 1 | **Hugging Face 계정에서 [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1) 모델 페이지의 사용 약관(user conditions)에 먼저 동의** | Community-1 은 gated 모델입니다. **약관 승인 전에는 토큰이 있어도 다운로드가 거부됩니다.** |
| 2 | [hf.co/settings/tokens](https://hf.co/settings/tokens) 에서 **read** 토큰 발급 → 프로젝트 루트 `.env` 에 `HF_TOKEN=hf_...` | 토큰은 `.env`/환경변수로만 읽습니다. 코드·설정·로그·출력물에 저장/출력하지 않습니다(`.env` 는 git-ignore). |
| 3 | (권장) 인터넷 가능한 PC 에서 `python scripts/prefetch_models.py` 1회 실행 | Whisper·정렬·pyannote·WeSpeaker·NLTK 데이터를 미리 캐시. 이후 `HF_HUB_OFFLINE=1` 로 **완전 오프라인 실행** 가능 |
| 4 | 화자별 **등록 음성 준비**: 1인당 3개 이상, 각 10–30초, 조용한 환경, 평소 말투, 회의와 같은 마이크면 더 좋음 | 등록 품질이 식별 정확도를 좌우 |
| 5 | 사내 실제 회의 1–3건에 대해 정답 RTTM 작성 후 `scripts/evaluate.py` → `scripts/calibrate_threshold.py` | threshold·매칭 모드 확정 (§7.5, §9) |
| 6 | 사내 사용 전 모델 라이선스 확인(법무) | Community-1 / Whisper / WeSpeaker 사전학습 모델(학습 데이터셋 라이선스를 따름) 각각의 모델 카드 확인. 기본 WeSpeaker 모델 `english`(VoxCeleb ResNet221-LM)는 WeSpeaker 문서상 CC BY 4.0 |

**사내망 주의:** WeSpeaker 허브 모델은 `modelscope.cn` 에서 내려받습니다. 차단된 경우 WeSpeaker 문서의 Hugging Face 미러(예: `Wespeaker/wespeaker-voxceleb-resnet221-LM`) 또는 wenet.org.cn 에서 체크포인트를 받아 **`avg_model.pt` + `config.yaml` 로 이름을 맞춘 폴더**를 만들고 `speaker_id.wespeaker_model: <폴더 경로>` 로 지정하세요. nltk 는 프록시 환경에서 `--allow-proxied-nltk` 를 명시해야 받을 수 있습니다(프록시를 신뢰하는 경우에만).

---

## 5. 실행

### 5.1 UI (Gradio)
```bash
python app/main.py                    # http://127.0.0.1:7860  (기본은 로컬 전용 바인딩)
python app/main.py --port 7870 --config config/local.yaml
# 또는:  meeting-transcriber ui
```
| 탭 | 기능 |
|---|---|
| 회의록 생성 | 파일 업로드, Language(ko/auto/…), Speaker Count Mode(AUTO/RANGE/FIXED), Exact/Min/Max, Whisper Model, 등록 화자 식별 ON/OFF, (고급) 매칭 모드·threshold, 단계별 진행률(1.Audio preprocessing → 6.Export), Transcript / Speaker list / Similarity matrix / TXT·JSON·CSV·SRT 다운로드 |
| 화자 등록 | 이름 + 음성 여러 개 등록(같은 이름 재등록 시 샘플 추가), 재등록(덮어쓰기), 원본 음성 보관 여부, 등록 목록(샘플 수·사용 구간·유효 음성), 삭제, 원본 음성만 삭제, 저장 음성으로 재계산 |
| 시스템 진단 | Device, CUDA 사용 가능 여부, GPU 이름·메모리, torch/CUDA/cuDNN/CTranslate2 버전, Whisper·pyannote·WeSpeaker·정렬 모델, HF 토큰 설정 여부(값은 표시 안 함), 최근 실행의 검출 화자 수·처리 시간·오디오 길이·**RTF**·단계별 시간 |

### 5.2 CLI
```bash
meeting-transcriber diagnostics
meeting-transcriber enroll --name 이용욱 s1.wav s2.wav s3.wav          # 같은 이름이면 샘플 추가
meeting-transcriber enroll --name 이용욱 --replace new1.wav new2.wav   # 재등록(덮어쓰기)
meeting-transcriber enroll-dir data/enroll_raw                         # data/enroll_raw/<이름>/*.wav 일괄
meeting-transcriber speakers
meeting-transcriber transcribe 회의.m4a --mode FIXED --num-speakers 5
meeting-transcriber transcribe 회의.mp4 --mode RANGE --min-speakers 2 --max-speakers 10 --model large-v3-turbo
meeting-transcriber transcribe 회의.wav --language auto --no-identify
meeting-transcriber delete-raw-audio 이용욱     # 원본 등록 음성만 삭제(임베딩 유지)
meeting-transcriber delete-speaker 이용욱       # 프로필 전체 삭제
```
출력: `outputs/<YYYYMMDD_HHMMSS>_<파일명>/<파일명>.{txt,json,csv,srt}`

---

## 6. 결과 형식

**TXT** (같은 화자의 1초 이내 연속 발화는 읽기 편하게 병합)
```
00:00:01 - 00:00:05
이용욱
오늘 회의를 시작하겠습니다.

00:00:09 - 00:00:11
홍길동  (동시발화: 김철수)
잠시만요.
```

**JSON** (주요 필드, 전체는 실행 결과 참조. 음성 임베딩 벡터는 포함하지 않음)
```jsonc
{
  "file": "회의.m4a", "language": "ko", "duration": 3605.2, "speaker_count": 4,
  "speakers": [{"cluster": "SPEAKER_00", "label": "이용욱", "status": "identified", "similarity": 0.71,
                "best_candidate": "이용욱", "margin": 0.33, "matched_via": "hungarian", "candidates": [...],
                "speech_duration": 812.4, "evidence_sec": 120.0, "low_evidence": false, "window_agreement": 0.95}],
  "segments": [{"start": 1.0, "end": 5.0, "speaker_cluster": "SPEAKER_00", "identified_speaker": "이용욱",
                "speaker_similarity": 0.71, "speaker_status": "identified", "text": "…",
                "overlap": false, "secondary_speakers": [], "asr_avg_logprob": -0.21, "word_confidence": 0.83,
                "words": [{"word": "오늘", "start": 1.02, "end": 1.31, "score": 0.91,
                           "speaker_cluster": "SPEAKER_00", "speaker": "이용욱", "overlap": false, "secondary_speakers": []}]}],
  "diarization": {"speaker_diarization": [...], "exclusive_speaker_diarization": [...], "overlap_regions": [...]},
  "speaker_identification": {"profiles": [...], "clusters": [...], "similarity_matrix": [[...]], "cooccurrence_sec": [[...]],
                             "threshold": 0.5, "margin": 0.05, "matching_mode": "flexible", "evidence": {...}},
  "models": {...}, "settings": {...}, "processing": {"timings_sec": {...}, "real_time_factor": 0.08}, "warnings": [...]
}
```
- `identified_speaker`: 등록자 이름 / `UNKNOWN_xx`(미등록·저신뢰) / `SPEAKER_xx`(식별 OFF 또는 등록자 없음) / `UNASSIGNED`(어느 화자 구간과도 겹치지 않는 단어)
- `speaker_status`: `identified | unknown | insufficient_audio | not_identified | unassigned`

**CSV** 컬럼(정확히 6개, Excel 한글 호환 UTF-8 BOM): `start_time, end_time, speaker, speaker_cluster, confidence, text`
`confidence` = 화자 귀속 신뢰도(매칭된 프로필과의 cosine). UNKNOWN 은 공란.

**SRT**: `[화자명] 발화` 형식 자막.

---

## 7. 화자 등록 · 식별 설계

### 7.1 등록(Enrollment) — robust profile
1. 각 샘플 → 16 kHz mono 변환 → peak 정규화 → **Silero VAD 로 무음/비음성 제거**(VAD 는 [-1,1] float 스케일에서 수행)
2. 유효 음성 < 3초 샘플은 사유와 함께 제외
3. 긴 샘플은 10초 단위 chunk 로 분할 → chunk 별 WeSpeaker 임베딩 → **L2 정규화**
4. **Leave-one-out outlier 제거**: 각 임베딩과 "나머지의 평균" 간 cosine 이 robust z-score(중앙값/MAD) < −3 이거나 0.2 미만이면 제외(최소 절반은 유지)
5. 남은 임베딩의 **평균 → 재정규화 = centroid**(프로필 벡터)

> 근거: 길이 정규화 임베딩 평균은 화자 검증의 표준적인 다중 세션 등록 방식으로, 단일 샘플의 채널·세션 편차를 줄입니다. 첫 샘플만 쓰거나 단순 평균만 쓰면 잘못된 샘플(다른 사람/음악/클리핑) 하나가 프로필을 오염시키므로 outlier 제거를 추가했습니다.

저장 구조(로컬 전용, git-ignore): `data/speakers/<speaker_id>/{profile.json, embeddings.npy, centroid.npy, samples/*.wav}`. 프로필에는 생성한 **임베딩 모델 ID** 가 기록되며, 다른 모델로 만든 프로필은 매칭에서 제외되고 재등록 안내가 표시됩니다(임베딩 공간이 다르므로).

### 7.2 클러스터 대표 임베딩
짧은 단일 구간으로 식별하지 않습니다. SPEAKER_xx 마다:
겹침 구간 제거 → **1.5초 이상** 비겹침 발화 우선(없으면 0.5초 이상으로 완화 + `low_evidence` 표시) → 10초 이하 window 분할 → RMS −45 dBFS 미만(무음성) 제외 → 긴 순서로 최대 20개/120초 → 각 window WeSpeaker 임베딩 → **길이 가중 평균**. 1초 미만이면 `insufficient_audio` → UNKNOWN.

### 7.3 Scoring
WeSpeaker 공식 recipe(`wespeaker/bin/score.py`)의 방식인 **cosine similarity**(선택적으로 cohort 평균 벡터 차감, `speaker_id.cohort_mean_path`)를 사용합니다. 점수는 **raw cosine [−1, 1]** 입니다. WeSpeaker CLI 의 `compute_similarity` 는 `(cos+1)/2` 로 [0,1] 변환한 값이므로 threshold 비교 시 혼동하지 마세요. AS-Norm/QMF 는 cohort 데이터가 필요해 PoC 범위에서 제외했습니다(확장 지점).

### 7.4 매칭 알고리즘 비교와 기본값
| 모드 | 방식 | 장점 | 단점 |
|---|---|---|---|
| `argmax` (비교용) | 클러스터마다 독립적으로 최고 유사도 | 단순, 분할 화자도 같은 이름 | **한 사람이 여러 클러스터에 중복 배정**(오식별 전파) |
| `strict_one_to_one` | Hungarian(전역 최적 1:1) | 중복 배정 원천 차단 | diarization 이 한 사람을 2개로 쪼개면 약한 쪽이 UNKNOWN |
| **`flexible` (기본)** | Hungarian 1:1 → 남은 클러스터는 ① threshold+0.05 이상, ② margin 충족, ③ 이미 배정된 같은 사람의 클러스터와 **동시 발화 2초 이하**일 때만 같은 이름 허용 | 중복 배정 억제 + 분할 화자 복구 | 조건이 늘어 파라미터 calibration 필요 |

공통 수락 조건: `similarity ≥ threshold` **그리고** `similarity − 최상위 대안 ≥ margin(0.05)`. 대안에서는 다른 클러스터가 전역 해에서 확실히 차지한 사람은 제외합니다. 기본값을 `flexible` 로 둔 이유: 한 사람이 동시에 두 클러스터로 말할 수는 없다는 물리적 제약(동시 발화 시간)을 이용해 1:1 강제의 단점(분할 화자 UNKNOWN 처리)과 argmax 의 단점(중복 배정)을 동시에 줄이기 위함입니다. **최종 기본값은 §9 평가 결과(실제 회의)로 확정**해야 하며, `scripts/evaluate.py` 는 세 모드를 같은 유사도 행렬에서 재계산해 나란히 보고합니다.

### 7.5 UNKNOWN 처리와 threshold — 환경별 calibration 필요
- 유사도 미달, 후보 간 차이(margin) 부족, 근거 음성 부족 → **억지로 이름을 붙이지 않고** `UNKNOWN_01, UNKNOWN_02 …`(첫 등장 순). UNKNOWN 끼리도 서로 다른 화자로 유지됩니다. (예: 등록 8명 + 게스트 2명 → 게스트는 각각 UNKNOWN_01/02)
- `speaker_id.match_threshold` 초기값 **0.50 은 공식 수치가 아닌 PoC 출발점**입니다. WeSpeaker 는 운영 threshold 를 공개하지 않으며(EER 는 데이터·모델·정규화에 따라 달라짐), 회의 원거리 마이크·한국어·짧은 발화 조건에서 점수 분포가 달라집니다. 오식별보다 UNKNOWN 이 낫다는 원칙으로 보수적으로 설정했습니다.
- **반드시 사내 환경에서 calibration** 하세요:
  ```bash
  python scripts/calibrate_threshold.py                                   # 등록 데이터 leave-one-out (낙관적 추정)
  python scripts/evaluate.py --manifest data/eval/manifest.yaml           # 정답 RTTM 있는 실제/합성 회의
  python scripts/calibrate_threshold.py --scores outputs/eval_*/similarity_scores.json --write config/local.yaml
  ```
  권장값 = max(EER threshold, FAR 1% threshold). `MT_CONFIG=config/local.yaml` 로 적용.

---

## 8. Overlap 처리
- `speaker_diarization`(겹침 보존)과 `exclusive_speaker_diarization`(한 시점 1화자)을 **모두 JSON 에 보존**합니다. 원본 diarization 은 삭제하지 않습니다.
- 단어의 대표 화자 = exclusive diarization 과의 최대 교집합(전사 귀속용).
- 단어 구간의 30% 이상이 겹침 영역이면 `overlap: true`, 그 시간에 함께 말한 다른 화자는 `secondary_speakers` 로 기록(단어·발화 단위). 확장 시 secondary 화자의 발화 복원(음원 분리)으로 연결 가능한 구조입니다.
- 클러스터 대표 임베딩에는 겹침 구간을 사용하지 않습니다.

---

## 9. 평가 · 10-Speaker 테스트 시나리오
| 영역 | 지표 | 구현 |
|---|---|---|
| STT | **CER**(한국어 기본, 공백 제외/포함 모두), WER, 치환·삭제·삽입 수 | jiwer |
| Diarization | **DER** 및 구성요소 **Miss / False Alarm / Speaker Confusion** (collar 0, 0.25) | pyannote.metrics |
| 화자 식별 | Identification Accuracy, **Unknown rejection**, **False Accept**(미등록자를 등록자로), Misidentification(다른 등록자로), **False Reject**(등록자를 UNKNOWN 으로), 시간 가중 정확도, **등록 화자별 confusion matrix**, 시간 기준 attribution error | `evaluation/metrics.py` |

시나리오 A(2명)·B(5명)·C(10명)·D(등록 8+미등록 2)·E(겹침)·F(짧은 맞장구 "네./맞습니다./잠시만요./아니요.")·G(비슷한 음색): `tests/scenarios/README.md`.
`scripts/make_synthetic_meeting.py` 는 **등록에 쓰지 않은** 개인별 발화로 정답 RTTM 이 자동 생성되는 합성 회의를 만들고(프리셋 A–G, 잡음 SNR, seed), `scripts/evaluate.py` 가 매니페스트 기반으로 일괄 측정합니다. 합성 회의는 실제보다 쉬우므로 상한값으로 해석하고 최종 판단은 실제 회의 녹음으로 합니다.

---

## 10. 설정
`config/default.yaml` (주석 포함) → `MT_CONFIG` 로 지정한 YAML → 환경변수 `MT_<SECTION>__<KEY>` 순으로 덮어씁니다.

| 키 | 기본값 | 설명 |
|---|---|---|
| `runtime.device` | `auto` | auto/cuda/cpu |
| `asr.model` | `large-v3` | `large-v3-turbo` 등 faster-whisper 이름 |
| `asr.compute_type` | `auto` | cuda→float16, cpu→int8. OOM 시 `int8_float16` |
| `asr.batch_size` | 16 | OOM 시 8/4 |
| `asr.language` | `ko` | `auto` = 자동 감지 |
| `diarization.mode` | `AUTO` | AUTO / RANGE / FIXED (인원 확실하면 FIXED 우선) |
| `diarization.min_speakers` / `max_speakers` | 2 / 10 | RANGE |
| `diarization.speaker_limit` | 10 | UI/API 허용 상한(하드코딩 아님, 늘릴 수 있음) |
| `speaker_id.wespeaker_model` | `english` | 허브 이름 또는 로컬 폴더 |
| `speaker_id.match_threshold` | 0.50 | raw cosine, **calibration 필요** |
| `speaker_id.match_margin` | 0.05 | |
| `speaker_id.matching_mode` | `flexible` | flexible / strict_one_to_one / argmax |
| `speaker_id.enrollment.keep_raw_audio` | true | false = 임베딩만 저장 |
| `paths.enrollment_dir` / `output_dir` / `model_cache_dir` | data/speakers / outputs / null | |
| `ui.server_name` | 127.0.0.1 | 로컬 전용 |
| `ui.brand.*` | 한미약품 / AI 회의록 / `#e12319` | 회사명·앱 제목·부제·배지·푸터·메인/보조 컬러·로고·파비콘 (`assets/brand/README.md`) |

**UI 브랜딩:** 한미약품 CI 적용 — 단색 CI Red(`#E12319`) 포인트 + 화이트 레이아웃 + 차콜 텍스트, 헤더에 레드 타원 엠블럼·흰색 이탤릭 워드마크 "Hanmi"(`assets/brand/hanmi_emblem.svg`, 공개 CI 구성 기준 재구성본), 브라우저 탭 아이콘 동일. 제목·색상·로고는 `ui.brand` 설정으로 변경 가능. 웹폰트 없이 시스템 한글 폰트만 사용해 외부 요청이 발생하지 않습니다.

---

## 11. GPU / CPU / 장시간 음성
- **GPU 우선**, GPU 가 없으면 자동 CPU 전환(`compute_type` int8). CPU 에서 large-v3 는 실시간보다 수 배 느릴 수 있어 실사용 비권장이며, CPU 에서는 `large-v3-turbo + int8` 을 권장합니다(진단 탭에 안내 표시).
- **CUDA OOM**: torch / CTranslate2 OOM 을 단계별로 감지해 앱이 죽지 않고 "어느 단계에서, 무엇을 낮출지"(batch_size, compute_type, 모델 크기, pyannote batch)를 안내합니다.
- **1–3시간 회의**: 단계별 모델 해제로 GPU 메모리 누적 방지. 3시간 16 kHz mono float32 오디오는 RAM 약 0.7 GB. 단어-화자 결합·구간 선택은 이진 탐색 기반으로, 3시간 규모(단어 약 2.8만 개, 턴 2.6천 개) 합성 데이터에서 각각 0.4초 / 0.2초에 처리됨을 확인했습니다.
- 진단 탭/JSON 에 단계별 처리 시간과 RTF(처리시간 ÷ 오디오 길이), GPU peak 메모리를 기록합니다.

---

## 12. 정확도에 영향을 미치는 요소와 한계
화자 분리와 식별 정확도는 다음 요소에 크게 좌우되며, 어떤 설정으로도 100% 를 보장할 수 없습니다.

| 요소 | 영향 |
|---|---|
| 마이크 거리(원거리 회의실 마이크) | 잔향·SNR 저하로 임베딩 품질 하락 → 유사도 하락 |
| 배경 소음 | 오탐(False Alarm), 임베딩 오염 |
| 겹침 발화 | 한 시점 1화자 할당의 한계, Confusion 증가 |
| 잔향(reverberation) | 화자 특징 번짐 |
| 짧은 발화("네", "맞습니다") | 임베딩 근거 부족 → 분리/귀속 오류 증가 |
| 비슷한 목소리(성별·연령·음색) | 클러스터 병합·오식별 |
| 녹음 비트레이트/코덱 | 저비트레이트 압축(통화 녹음 등)은 고주파 손실 |
| 마이크 채널 구성 | 다채널을 downmix 하면 위상 간섭, 채널별 화자 정보 소실 가능(`audio.channel_strategy`) |

**본 구조의 개선 원리:** 일반 diarization 은 익명 클러스터만 제공하지만, 고정 참석자의 음성을 사전 등록(voice enrollment)하면 ① 클러스터에 이름을 자동 부여하고 ② 확신이 없는 경우 UNKNOWN 으로 분리하므로, 사람이 매번 SPEAKER_xx 를 수작업 매핑하던 방식보다 **화자 귀속(speaker attribution) 정확도와 일관성**을 높일 수 있습니다. 단, diarization 단계의 오류(분할/병합)는 식별 단계에서 일부만 보정됩니다(flexible 모드의 분할 화자 복구).

**Known limitations**
- threshold·매칭 모드 기본값은 사내 데이터로 검증되지 않았습니다(§7.5).
- 기본 WeSpeaker 모델(VoxCeleb)은 영어 위주 학습 데이터 → 한국어 회의 환경에서의 성능 미검증. `vblinkf`(VoxBlink2 다국어)를 라이선스 확인 후 비교 평가 권장.
- 겹침 구간에서는 단어당 대표 화자 1명만 텍스트를 가지며, 동시에 말한 두 번째 화자의 발화 내용은 복원하지 않습니다(secondary_speakers 표시만).
- 동일 인물의 다중 클러스터 분할을 flexible 모드가 일부 복구하지만, 서로 다른 두 사람이 하나의 클러스터로 병합된 경우는 식별 단계에서 분리할 수 없습니다(FIXED 모드로 인원 지정 권장).
- 한국어 alignment 모델(`kresnik/wav2vec2-large-xlsr-korean`)의 사전에 없는 문자(숫자·영문 등)는 WhisperX 가 wildcard/보간으로 처리 → 해당 단어 타임스탬프 정밀도 저하.
- AS-Norm / score calibration(QMF) 미구현.

---

## 13. 보안 · 개인정보(음성정보)
| 원칙 | 적용 |
|---|---|
| HF 토큰 | `.env`/환경변수만 사용, `.env` git-ignore, 로그 필터가 토큰 값·`hf_...` 패턴 마스킹, 진단에는 설정 여부만 표시 |
| 외부 전송 기본 OFF | 모든 추론 로컬. pyannote 텔레메트리(`PYANNOTE_METRICS_ENABLED=0`), HF Hub 텔레메트리, Gradio analytics 기본 비활성. Gradio `share=False`(외부 터널 금지), `127.0.0.1` 바인딩. pyannoteAI 유료 API 미사용 |
| Local storage | 프로필·등록 음성·결과물은 로컬 디스크(`data/`, `outputs/`)에만 저장, git-ignore(`*.wav`, `*.npy` 포함) |
| 원본 등록 음성 삭제 | UI "원본 음성만 삭제" / CLI `delete-raw-audio` / `keep_raw_audio=false` |
| Voice profile 삭제 | UI "Delete" / CLI `delete-speaker` → 해당 폴더 전체 삭제 |
| 로그 | 임베딩·음성 데이터 dump 금지. JSON 결과에도 임베딩 벡터 미포함(유사도 점수만) |
| 원본 파일 | 읽기 전용 취급, 작업 사본은 처리 후 삭제(`audio.keep_intermediate_wav=false`) |

Voice embedding 은 생체정보에 준하는 개인정보로 취급하십시오. 운영 전환 시 등록 동의서, 보관 기간, 접근권한(RBAC), 감사 로그 정책을 별도로 수립해야 합니다.

---

## 14. 검증 현황
### 14.1 실제로 실행한 검증 (이 저장소의 개발 환경: Linux x86_64, Python 3.11, **GPU 없음**, Hugging Face·ModelScope·PyTorch 인덱스 **네트워크 차단**)
| 항목 | 결과 |
|---|---|
| 설치(uv / pip 해석기) | 성공, `uv pip check`: 166 패키지 호환, 충돌 없음 |
| import / syntax | 전 모듈 import 성공, `compileall` 통과, ruff(E,F,W,B) 통과 |
| `pytest` | **109 passed, 1 skipped**(실모델 E2E opt-in) |
| upstream API 회귀 가드 | whisperx 3.8.6 / pyannote.audio 4.0.7 / WeSpeaker / faster-whisper 실제 설치본 시그니처 일치 |
| pyannote 반환 객체 | 실제 `DiarizeOutput` 클래스로 변환·두 diarization 보존·overlap 계산 검증 |
| WhisperX alignment | 실제 `whisperx.load_align_model` + `whisperx.align` 코드 경로를 **로컬 소형 랜덤 Wav2Vec2-CTC(한국어 문자 사전)** 로 실행 → 단어 타임스탬프 생성 → 화자 결합 → 4개 형식 export |
| WeSpeaker | 실제 `wespeaker.load_model`(로컬 폴더) + kaldi fbank + 추론 경로를 **랜덤 가중치 ResNet34** 로 실행 → 등록/추가/재등록/재계산/원본삭제/삭제, 식별 단계 전체 데이터 흐름 |
| 실음성 | pyannote 내장 30초 실음성 샘플로 Silero VAD 등록 경로, DER 계산, 합성 회의 생성기 검증 |
| 오디오 입력 | WAV/MP3/M4A/MP4/MOV → 16 kHz mono 변환, 원본 해시 불변, 손상 파일·무음 영상·미지원 확장자 오류 처리 |
| UI | Gradio 실제 기동 + `gradio_client` 로 업로드/실행/등록/진단 API 호출, 3개 탭 구성 확인 |
| 오류 처리 | 모델 다운로드 실패(프록시 403 → 네트워크로 정확 분류), CUDA 미존재, 잘못된 화자 수 설정, 손상 입력 → 단계명 포함 사용자 메시지(트레이스백 노출 없음) |
| 장시간 | 3시간 규모 합성 타임라인에서 결합·구간선택 성능 확인 |

### 14.2 미검증 항목 (모델 다운로드 · HF 인증 · GPU · 실제 한국어 음성이 없어 실행 불가)
- 실제 Whisper `large-v3`/`large-v3-turbo` 한국어 전사 품질(CER)
- 실제 `kresnik/wav2vec2-large-xlsr-korean` 정렬 정확도
- 실제 pyannote Community-1 다운로드(HF 토큰·약관 승인)와 diarization 실행, AUTO/RANGE/FIXED 결과
- 실제 WeSpeaker 사전학습 모델(`english`/`vblinkf`) 다운로드와 한국어 화자 식별 정확도, threshold 적정성
- CUDA 실행, VRAM 사용량, OOM 실제 발생 시 동작, RTF 수치
- Windows 설치 절차(문서 기준 작성, 실제 Windows 미검증)
- 1–3시간 실제 녹음의 메모리/시간

검증 명령(모델·토큰·GPU 준비 후):
```bash
RUN_INTEGRATION=1 MT_TEST_AUDIO=회의.wav MT_TEST_NUM_SPEAKERS=3 MT_TEST_ENROLL_DIR=data/enroll_raw \
  pytest -m integration tests/integration/test_full_pipeline.py -s
```

---

## 15. 트러블슈팅
| 증상 | 조치 |
|---|---|
| `pyannote 모델 접근 권한이 없습니다` / 파이프라인을 내려받지 못함 | Community-1 페이지 약관 동의 여부, `.env` 의 `HF_TOKEN`(read 권한, `hf_` 로 시작) 확인 |
| `다운로드 실패(네트워크/프록시 차단)` | 사내망 HF 접근 확인 → 가능한 PC 에서 `prefetch_models.py` 후 캐시 복사 + `HF_HUB_OFFLINE=1` |
| `NLTK 'punkt_tab' 데이터가 없고…` | `python scripts/prefetch_models.py --skip whisper align pyannote wespeaker --allow-proxied-nltk` (프록시 신뢰 시) 또는 `nltk_data` 폴더 복사 |
| WeSpeaker 다운로드 실패 | modelscope.cn 차단 → §4 수동 설치(폴더에 `avg_model.pt` + `config.yaml`) |
| `Could not load library libcudnn_ops.so.9` 등 | CTranslate2 4.x 는 CUDA 12 + cuDNN 9 필요. Linux: `export LD_LIBRARY_PATH=$(python3 -c 'import os, nvidia.cublas, nvidia.cudnn; print(os.path.dirname(nvidia.cublas.__path__[0]) + "/cublas/lib:" + os.path.dirname(nvidia.cudnn.__path__[0]) + "/cudnn/lib")')` (faster-whisper README 명령) |
| CUDA out of memory | `MT_ASR__BATCH_SIZE=8`, `MT_ASR__COMPUTE_TYPE=int8_float16`, `MT_ASR__MODEL=large-v3-turbo`, `MT_DIARIZATION__EMBEDDING_BATCH_SIZE=8` |
| 모두 UNKNOWN | threshold 과도 → calibration. 등록 샘플 품질/길이, 회의 마이크와 등록 마이크 차이 확인 |
| 한 사람이 여러 이름 | `strict_one_to_one` 시험, FIXED 모드로 인원 지정, 등록 샘플 추가 |
| 프로필이 매칭에서 제외됨 | 임베딩 모델 변경 → "저장 음성으로 재계산" 또는 재등록 |

---

## 16. 프로젝트 구조
```
app/main.py                         Gradio 진입점
assets/brand/                       로고·파비콘 배치 위치(안내 README)
config/default.yaml                 전체 설정(주석)
src/meeting_transcriber/
  config.py  device.py  errors.py  logging_utils.py  cli.py  ui.py  branding.py(테마·헤더·푸터)
  audio/          io.py(ffprobe/ffmpeg)  preprocess.py(정규화, noise hook)
  asr/            whisperx_asr.py(전사 + 정렬)
  diarization/    pyannote_diarizer.py  types.py(overlap/exclusive 결과 모델)
  speaker_id/     embedder.py(WeSpeaker)  profiles.py(등록 저장소)  cluster_embedding.py
                  scoring.py  matching.py  identify.py
  pipeline/       runner.py(오케스트레이션)  assignment.py(단어-화자 결합)
  export/         writers.py(TXT/JSON/CSV/SRT)
  evaluation/     metrics.py  calibration.py  report.py  synthetic.py
scripts/          prefetch_models.py  evaluate.py  calibrate_threshold.py  make_synthetic_meeting.py
tests/unit, tests/integration, tests/scenarios(A–G 매니페스트·가이드)
tests/local_models.py               오프라인 E2E용 로컬 모델 생성기(실제 구조·랜덤 가중치)
data/speakers/    (git-ignore) 음성 프로필
outputs/          (git-ignore) 결과물
```
