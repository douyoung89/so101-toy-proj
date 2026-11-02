# 데이터 수집과 전체 실험

루트 [README](../README.md)에서 데모와 소형 특징 학습부터 시작할 수 있습니다. 이 폴더는 23개 손목 각도의 관측표, 여러 수집 전략의 에피소드, 인과적 정책 입력을 만드는 파이프라인입니다.

## 새 데이터 수집

저장소 루트에서 `requirements-sim.txt` 설치와 `python scripts/setup_assets.py --ocr`를 먼저 실행합니다. 원본 `partial_v1`의 대량 이미지는 공개본에 포함하지 않으므로 **새 데이터 경로**를 사용합니다. 아래 명령은 macOS 그래픽 접근 가능한 터미널에서 실행합니다.

```bash
cd so101
python partial_scene.py init --data data/my-pilot --scenes 15
python partial_scene.py collect --data data/my-pilot
python check_partial_scene.py --data data/my-pilot
python sample_partial_rollouts.py --data data/my-pilot
python collect_partial.py --data data/my-pilot
python audit_partial.py --data data/my-pilot --reproduce
python build_policy_dataset.py --data data/my-pilot
python evaluate_partial.py --data data/my-pilot --development
```

장면당 23개 각도(−70°~+40°, 5° 간격)의 원본 960×720 및 파생 224×224 영상과 OCR·관절 상태를 저장합니다. 좌측 우선·우측 우선·무작위·최단 수집 경로가 시작점을 공유합니다. 관측표는 미래를 보는 수집용 oracle에 사용하며, 온라인 정책에는 현재까지의 정보만 전달합니다.

시작 기억이 불완전하고 안전한 이동으로 8회 안에 두 값을 정확히 얻을 수 있는 시작점이 있어야 장면을 채택합니다. 거절과 실패도 보존합니다. 완료한 장면은 해시를 검사한 뒤 건너뛰며, 코드·설정 변경 후에는 새 데이터 경로를 사용합니다.

`audit_partial.py --reproduce`는 물리 상태를 재현합니다. 픽셀 동일성을 검증한다는 뜻은 아닙니다. `policy/*`의 input만 모델 입력으로 전달하며 metadata·진실값·미래 관측은 입력에서 제외합니다.

## 새 데이터에서 방향 모델 실험

```bash
cd experiments/exp02_structured
python run_experiment.py prepare --data ../../data/my-pilot --run runs/my-experiment
python run_experiment.py train --run runs/my-experiment
python run_experiment.py evaluate --run runs/my-experiment
python run_experiment.py report --run runs/my-experiment
python run_experiment.py integrity --run runs/my-experiment
```

`prepare`는 원본 이미지와 해시·기억 재생을 감사하고 교사 방향을 재생성합니다. train/val 방문 OCR과 관측표의 accepted 값·모호성 상태가 모두 일치해야 진행합니다. 새 수집의 OCR이 달라지면 이 신뢰성 검사가 실패할 수 있습니다.

`evaluate`는 checkpoint 선택 후 새 영상으로 val/test를 비교합니다. 선택 후 재학습을 차단하고 완료된 test 사례를 재사용합니다. 공개된 라벨은 개발에 사용되었으므로 새로 촬영해도 미사용 라벨의 최종 평가로 해석하면 안 됩니다.

## 파일과 출처

- `common.py`: 저장·해시·정책 입력 허용 필드·지표.
- `simulator.py`: `../support/`의 관절·투영·접촉·OCR 함수를 재사용. `../assets/reference/`의 기준 형상을 읽고 현재 경로로 XML을 생성.
- `partial_scene.py` → `check_partial_scene.py` → `sample_partial_rollouts.py` → `collect_partial.py`: 장면 생성·검수·시작점 선택·실제 수집.
- `audit_partial.py`, `build_policy_dataset.py`, `evaluate_partial.py`: 검수·인과적 입력 내보내기·온라인 평가.
- `experiments/exp02_structured/`: 구조화 상태 방향 MLP.
- `examples/right_first_policy.py`: 이미지 접근과 외부 콜백 연결을 보여주는 미학습 규칙 예시.

기준 가림판 크기는 0.056 × 0.0096 × 0.0012m, 기준 손목 명령값은 0.75rad입니다. 원래 무제한 선형 연동에 새 중심 이동 한계와 포화 연동을 추가했습니다. 매 물리 스텝에서 `clip(center + slope × (wrist-reference), limits)`로 soft equality 목표를 갱신합니다. 이는 실물 기구를 검증한 결과가 아닙니다.

[원 설계](DESIGN.md)와 [과거 수집 보고서](PILOT_REPORT.md)는 설계·개발 기록입니다. 과거 보고서의 이미지·데이터 경로 중 공개본에 없는 항목은 [배포 안내](../docs/PUBLICATION.md)를 참고하세요.
