# 공개 사본 구성

원본 실험 디렉터리는 변경하지 않고 필요한 파일만 별도 저장소로 복사했습니다.

| 원본의 역할 | 공개 사본 |
|---|---|
| partial v1 수집·감사·평가 코드 | `so101/` |
| 최근 구조화 상태 MLP 실험 | `so101/experiments/exp02_structured/` |
| 초기 실험의 공통 함수 | `support/stage2.py`, `stage3.py`, `stage5.py`, `wrist_sweep.py` |
| 고정 v2_005 장면과 OCR 임계값 | `assets/reference/` |
| train/val/test 특징·제외 로그·체크포인트 | `results/exp02/` |
| 새 영상 온라인 평가 15건의 텍스트 기록 | `results/exp02/evaluation/` |
| val scene_000008 시연 | `docs/media/`와 `assets/demo/` |
| 원래 100개 합성 라벨 | `data/labels/images.zip`과 CSV |

## 제외한 자료

전체 `so101/`는 원본에서 약 239MB였으며, 상위 폴더에는 별도로 약 1.25GB의 출력과 약 478MB의 외부 저장소가 있었습니다. 공개본에서는 대량 관측 PNG·거절 후보 이미지·중복 실험·EasyOCR 가중치·외부 mesh checkout·캐시·소스 스냅샷을 제외했습니다. 원본은 보존되어 있습니다.

`results/`는 선별된 과거 실행 증거입니다. 거기 적힌 이미지 경로의 실제 파일은 포함하지 않습니다. 전체 이미지 해시 감사, 수집 당시 소스 스냅샷 검증, 픽셀 재현은 이 사본만으로 수행할 수 없습니다. `so101/PILOT_REPORT.md`의 과거 `data/partial_v1` 링크는 공개 사본 위치에 맞춰 수정했습니다.

개인 컴퓨터 절대 경로는 `$ORIGINAL_PROJECT`·`<original-home>`으로 치환했습니다. 이 때문에 과거 JSON/텍스트 파일의 바이트 해시와 `selection.json`이 기록한 원래 experiment 파일 해시는 공개본과 달라질 수 있습니다. 체크포인트와 구조화 특징 행은 바꾸지 않았으며 별도로 검증합니다.

## 경로·실행 변경

재사용 모듈을 `support/`로 옮겨 경로 상수를 조정했고, 기준 XML·임계값을 `assets/reference/`로 분리했습니다. 새 장면 XML은 현재 저장소의 mesh·라벨 경로로 생성합니다. 공개 학습·시연용 스크립트는 새 `runtime/`에만 출력하고 과거 결과를 덮어쓰지 않습니다.

구조화 특징 학습에는 원본 이미지가 필요하지 않습니다. 새 교사 재생성과 전체 감사에는 새 데이터를 먼저 수집해야 합니다. 라이브 데모는 보존한 val 설정에서 새 시뮬레이션 XML을 만들며 원본 test를 재실행하지 않습니다.

`assets/publication-manifest.json`은 선별한 공개 파일의 SHA-256 목록입니다. `scripts/project.py verify`가 이를 검사합니다. 새로 수집한 run에서는 기존 source_inventory 검사가 별도로 적용됩니다.
