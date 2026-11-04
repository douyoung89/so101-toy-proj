# 원본 관측 이미지

원본 PNG 356장을 `images_01.zip`부터 `images_08.zip`까지 나누어 보관했습니다. GitHub API의 단일 파일 전송 제한 때문에 압축 파일로 배포합니다.

프로젝트 루트에서 `python scripts/setup_assets.py` 또는 `./run_experiment.sh`를 실행하면 이 폴더에 `frame_000001.png` 등의 원본이 자동으로 풀립니다. 에피소드 JSON의 `data/images/frame_*.png` 경로는 압축 해제 후 그대로 사용할 수 있습니다.

이미지 자체와 에피소드의 관측 순서는 바꾸지 않았습니다. 동일 이미지 파일을 여러 관측에서 참조하는 방식도 유지합니다.
