"""Build a concise Korean report from recorded artifacts, without rerunning test."""
from common import read, write


def report(run):
    inventory = read(run/'inventory.json')
    selection = read(run/'selection.json')
    results = read(run/'evaluation_report.json')
    summaries = {s:read(run/'data'/f'{s}_summary.json') for s in ['train','val','test']}
    schema = read(run/'feature_schema.json')
    features_table = '\n'.join(f'| {s} | {inventory["scenes"][s]} | {len(inventory["label_ids"][s])} | '
        f'{inventory["episodes"][s]} | {inventory["original_policy_rows"][s]} | {summaries[s]["retained_unique_states"]} |'
        for s in ['train','val','test'])
    label_table = '\n'.join(f'| {s} | '+ ' | '.join(str(summaries[s].get(k,0)) for k in
         ['unique','tie','complete_correct','complete_wrong','budget_exhausted','unsolvable','duplicate_states_collapsed'])+' |'
         for s in summaries)
    metrics_table = []
    for split, methods in results['live'].items():
        for method, r in methods.items():
            m = r['valid_cases']
            count = m['episodes']
            success = round(m['exact_acquisition_rate']*count) if count else 0
            wrong = round(m['wrong_stop_rate']*count) if count else 0
            metrics_table.append(f'| {split} | {method} | {success}/{count} | {wrong}/{count} | '
                f'{m["mean_moves"]:.2f} | {r["termination_counts"].get("budget_exhausted",0)} | {r["technical_errors"]} |')
    classifiers = {'train':selection['train'], 'val':selection['val'], 'test':results['offline']['test']}
    classification_table = '\n'.join(f'| {s} | {m["states"]} | {m["optimal_direction_accuracy"]:.1%} | {m["soft_ce"]:.4f} |'
                                       for s,m in classifiers.items())
    failures = []
    for split, methods in results['live'].items():
        for method,r in methods.items():
            for f in r['failures']:
                failures.append(f'- {split} / {f["scene_id"]} / {method}: {f["reason"]}, '
                    f'{f["moves"]}회 이동, 잘못된 STOP={f["wrong_stop"]}, OCR 오독 {len(f["posthoc"]["ocr_misreads"])}건.')
    agreement = inventory['table_agreement_train_val']
    preservation = read(run/'preservation_after_evaluation.json')
    text = f'''# 실험 2: 구조화된 상태 입력을 사용하는 이동 방향 정책

77차원 상태를 입력받는 작은 2층 MLP를 맥북 CPU에서 학습하고, 새 MuJoCo 영상과 OCR로 평가했습니다.
이 자료는 개발 파일럿이며 최종 일반화 성능을 주장하지 않습니다. 기존 test 라벨은 이전 실험에서 이미 개발에 사용됐습니다.

## 데이터와 교사 라벨

원본은 장면 15개 / 관측표 345건 / 에피소드 60개로 재확인했습니다.

| 분할 | 장면 | 라벨 ID | 에피소드 | 기존 판단 시점 | 재라벨·중복 제거 후 이동 상태 |
|---|---:|---:|---:|---:|---:|
{features_table}

기존의 수집 행동을 정답으로 복사하지 않았습니다. 각 시점의 **실제 누적 OCR 기억**, 현재 각도,
남은 이동 횟수에서 출발해 관측표의 OCR과 기존 InformationMemory 규칙으로 좌우 두 첫 행동의 최단 해결 거리를 BFS로 계산했습니다.
정확한 두 값 확보는 오프라인 정답으로 판단하며, 이 정답은 교사 라벨 생성·사후 점수 계산에만 사용합니다.
미래 OCR은 교사 검색에만 사용하고 정책 입력에는 넣지 않습니다.

- 짧은 해결 경로의 첫 방향을 선택합니다. 최단 거리가 같으면 두 방향에 0.5씩 주는 soft CE를 사용합니다. 분류 정확도에서는 둘 다 정답입니다.
- 오독된 현재 기억을 정답으로 바꾸거나 초기화하지 않습니다. 충돌 해소에 필요한 후속 2회 확인을 그대로 모사합니다.
- 두 값이 모두 확인된 기억은 즉시 종료합니다. 값이 틀려도 교사가 가상의 복구 이동을 추가하지 않습니다.
- 예산 0, 이미 완료된 상태, 남은 예산에서 해결 불가인 상태에는 임의 방향을 붙이지 않고 제외 로그에 보존합니다.
- 같은 장면·같은 상태 특징·같은 교사 라벨의 중복은 묶고 원본 참조를 모두 보존합니다. 다른 라벨 ID의 파생 자료는 원 분할을 유지합니다.
- train/val 방문 {agreement['observations']}건 중 {agreement['matching']}건에서 관측표와 실제 방문의 accepted 값·모호성 상태가 일치했습니다.
  따라서 재라벨링을 채택했습니다. 이는 이 방문들에 대한 확인이며 모든 미래 경로의 OCR 동일성을 보장하지 않습니다.

아래 수는 중복 제거 전 전체 기록 관측을 분류한 것이며, 제외는 파일 삭제를 뜻하지 않습니다.

| 분할 | 단일 교사 방향 | 동률 | 정확히 완료 | 오독으로 완료 | 예산 소진 | 해결 불가 | 병합 중복 |
|---|---:|---:|---:|---:|---:|---:|---:|
{label_table}

## 상태 특징과 모델

- 현재 실제 손목 각도를 고정 기준 0.75 rad에 대한 상대각으로 변환하고 /70 정규화합니다. 남은 예산은 /8입니다.
- 단백질·당류 각각 unknown/confirmed/conflict 3상태, 누적 확인 신뢰도, 충돌 후 재확인 횟수를 사용합니다.
- 23개 각도의 과거 방문 횟수, 최대 9개의 실제 방문 각도 순서와 padding mask, 최대 8개의 **과거** 좌우 행동과 mask를 사용합니다.
- 각도는 0.001° 단위, 신뢰도는 소수점 4자리로 양자화합니다. OCR 숫자값과 원문·이미지·파일 경로는 입력에서 제외했습니다.
- 실제 영양 정답, 가림판 설정, 미래 관측, oracle 경로, 수집 전략명, 장면·라벨 ID도 모델 입력에 없습니다.
- 입력 {schema['dimension']} → Linear(32) → ReLU → Linear(2), 파라미터 {selection['parameters']:,}개. 출력은 LEFT_5/RIGHT_5뿐입니다.
- STOP은 기존 기억의 두 confirmed 값 확보 규칙으로 처리하고, 예산 소진 종료와 별도로 기록합니다. 범위 밖 방향은 마스킹하고 실행 전 기계·장면 안전 한계를 다시 확인합니다.

AdamW(lr=0.003, weight_decay=0.001), full batch, 최대 400 epoch, patience 60, CPU 1 thread,
고정 seed={selection['seed']}입니다. 특징 평균·표준편차는 train만으로 계산했습니다.
val soft CE가 가장 작은 epoch {selection['selected_epoch']}를 선택했습니다({selection['epochs_run']} epoch 실행).
학습 시간은 {selection['elapsed_seconds']:.2f}초입니다. actor–critic과 영상 인코더는 없습니다.

## 분류 결과와 실제 이동 평가

| 분할 | 평가 이동 상태 | 교사 최적 방향 정확도 | soft CE |
|---|---:|---:|---:|
{classification_table}

선택 체크포인트를 잠근 뒤에만 test 분류와 새 영상 에피소드 평가를 실행했습니다.
val 온라인 결과도 체크포인트 선택 이후에 생성했으며 추가 튜닝에 사용하지 않았습니다.
완료된 test 사례는 재실행 시 다시 촬영하지 않고 기록을 읽습니다.

각 장면의 기존 시작점·seed를 공유해 MLP와 좌측 우선·우측 우선·무작위·가까운 끝부터 고정 순회를 비교했습니다.
매 행동 뒤 실제 MuJoCo 이동 → 새 촬영 → 새 OCR → 기억 갱신을 실행했습니다.
관측표·정답은 정책 행동 선택에 사용하지 않았고, 모든 영상과 전이는 이 run 폴더 안에 새로 저장했습니다.

| 분할 | 정책 | 정확한 두 값 확보 | 잘못된 STOP | 평균 이동 | 예산 소진 | 기술 오류 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(metrics_table)}

이동량은 시작 자세로 옮기는 이동을 제외하며 성공·실패를 모두 포함합니다.
기술 오류와 탐색 실패·OCR 오독을 구분하고 원본 에피소드를 보존합니다.

## 실패 사례와 해석 범위

{chr(10).join(failures) if failures else '이번 작은 평가에서 실패 사례가 없었습니다.'}

분할은 train 라벨 12개 / val 2개 / test 1개입니다. 온라인 test는 **1개 장면의 1개 시작점**이므로
정책 순위나 새 라벨·가림판에 대한 일반화 결론을 낼 수 없습니다.
관측표 기반 교사는 이상화된 OCR 전이를 가정합니다. 온라인에서는 경로·물리 오차에 따라 OCR이 달라질 수 있습니다.
상태에 숨은 가림판의 연동 방향이 없고 이미지도 없으므로 같은 상태에서 다른 장면의 교사 방향이 충돌할 수 있습니다.
특징 충돌 통계는 `feature_schema.json`에 보존했습니다. 실제 분류·탐색 결과를 성공만 추려 재작성하지 않았습니다.

원본 파일 {preservation['checked_files']}개의 전후 SHA-256이 일치했습니다. 기존 파일럿·원 실험 소스는 수정하지 않았습니다.

## 재현

맥북의 기존 `so101` Conda 환경에서 충분합니다. 이 모델에는 GPU나 코랩이 필요하지 않습니다.
실제 영상 평가는 macOS 그래픽 컨텍스트를 사용할 수 있는 로컬 터미널이 필요합니다.

```bash
conda activate so101
cd so101/experiments/exp02_structured
python run_experiment.py prepare --run runs/reproduction
python run_experiment.py train --run runs/reproduction
python run_experiment.py evaluate --run runs/reproduction
python run_experiment.py report --run runs/reproduction
python -m unittest discover -s tests -v
```

기존 run은 덮어쓰지 않습니다. test 시작 후 재학습은 차단합니다. 새 seed나 모델 비교를 할 때는 새 run을 만들고,
이미 확인한 이 test를 새로운 최종 홀드아웃처럼 해석하지 않아야 합니다.

`best.pt`, `selection.json`, `training_history.jsonl`, `data/*_summary.json`, `evaluation_report.json`,
`evaluation/*/scene_*/<method>/episode.json`, `source_inventory.json`, `source_snapshot/`이 재현 근거입니다.
'''
    (run/'REPORT.md').write_text(text)
    return run/'REPORT.md'
