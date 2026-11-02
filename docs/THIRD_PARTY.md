# 기반 소프트웨어와 자산

- **TheRobotStudio / SO-ARM100**: https://github.com/TheRobotStudio/SO-ARM100 — 기준 커밋 `a758567c3978dfeefe282ede0500085a48fe8f78`, Apache License 2.0. [라이선스 사본](licenses/SO-ARM100-Apache-2.0.txt).
  `assets/reference/robot.xml`과 `scene.xml`은 SO101 모델을 바탕으로 수정했습니다. 라벨 상자, 마운트 방향, hood·가림판·연동, 접촉 제외, 고정 라벨 카메라 및 시각 설정을 추가·변경했으며 이번 공개 사본에서는 로컬 절대 경로를 상대 경로로 바꿨습니다. mesh는 재배포하지 않고 고정 커밋을 별도로 다운로드합니다. 기존 파일의 저작권 표시는 유지합니다.
- **MuJoCo**: https://github.com/google-deepmind/mujoco — 시뮬레이션·렌더링.
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR — 영어 OCR. 가중치는 저장소에서 제외하고 선택적 설치 단계에서 다운로드합니다.
- **PyTorch**: https://github.com/pytorch/pytorch — 작은 방향 선택 MLP.

데모 영상은 이 프로젝트의 시뮬레이션 실행 기록입니다. 합성 라벨 PNG는 기존 프로젝트가 생성한 자산입니다. 운영체제 폰트 파일은 재배포하지 않습니다. 이 문서는 프로젝트 자체에 새로운 라이선스를 부여하지 않습니다.
