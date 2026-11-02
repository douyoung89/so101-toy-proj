# 구조화된 상태 기반 이동 방향 MLP

[루트 실행 안내](../../../README.md) · [결과 보고서](../../../results/exp02/REPORT.md) · [보존 자료 설명](../../../results/exp02/README.md)

1. `core.py`: 과거 정보만 사용하는 77차원 특징과 실제 OCR 기억에서 출발하는 budgeted BFS 교사.
2. `prepare.py`: 분할·해시·기억 재생 감사, 재라벨링과 중복 병합.
3. `model.py`: Linear(77,32) → ReLU → Linear(32,2).
4. `train.py`: train 정규화, val soft CE로 체크포인트 선택, 선택 후 재학습 차단.
5. `evaluate.py`: MLP·좌측 우선·우측 우선·무작위·고정 순회의 새 촬영 평가.
6. `report.py`: 저장된 기록에서 보고서 생성.
7. `demo/watch_policy.py`: 학습한 모델이 현재 상태에서 결정하는 연속 움직임 촬영.

공개본의 소형 특징 재학습과 시연은 루트 `scripts/project.py`로 실행합니다. 원본 이미지 감사와 새 교사 생성은 `so101/README.md`의 새 데이터 수집 순서를 따릅니다. 과거 run 전체를 포함하지 않으므로 기본 `runs/pilot_20261102`는 없습니다.

train/val/test 특징은 81/14/5개입니다. 방향 예측만 학습하고 STOP은 기억 규칙이 처리합니다. 작은 val/test의 교사 방향이 모두 오른쪽이므로 규칙 대비 우위와 양방향 일반화를 입증하지 못했습니다.
