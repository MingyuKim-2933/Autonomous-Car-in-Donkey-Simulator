# Autonomous Car in Donkey Simulator

(사)한국인공지능연구소 AI 오픈랩 8기에서 진행한 프로젝트입니다. DonkeyCar 시뮬레이터에서 강화학습으로 자율주행 정책을 학습했습니다.

- 주간 기록: https://www.ai-lab.kr/opens/60125af9d85048992b7efa12
- 환경 구축과 진행 기록(Notion): https://www.notion.so/Self-Driving-Car-75187046f2bb48e89fef23e1446eff63
- 최종 발표: https://www.youtube.com/watch?v=J0vdPe0Jf1o
- SAC + VAE 최종 주행 영상: https://www.youtube.com/watch?v=rLE7WS1Hor4

## 결과

5가지 학습 방식을 같은 시뮬레이터에서 비교했고, **SAC + VAE**가 가장 빠르게 완주했습니다.

| 방식 | 완주까지 걸린 Episode |
|---|---|
| DDQN | 약 4,000 |
| DDQN + VAE | 약 900 |
| SAC + VAE | 300 |

## 구현 상세

### 공통
- 스로틀은 고정하고 조향만 학습합니다.
- 트랙을 벗어나 에피소드가 끝나면 보상 -10을 줍니다.

### DQN, DDQN (`train_dqn.py`, `train_ddqn.py`)
- 카메라 영상을 그레이로 바꾸고 80×80으로 줄인 뒤 4프레임을 쌓아 CNN 입력으로 씁니다.
- 조향 값을 구간으로 나눈 이산 행동으로 학습합니다.
- 학습률 1e-4, ε을 1.0에서 0.02까지 줄이며 탐색하고, 타깃 네트워크를 따로 둡니다.

### VAE 전처리 (`train_ddqn_vae.py`, `train_sac_vae.py`)
- 80×160 영상을 미리 학습한 VAE 인코더(`vae/vae-level-0-dim-32.pkl`)로 32차원 잠재 벡터로 압축해 상태로 씁니다.
- 큰 이미지 대신 작은 벡터로 학습해서 학습 속도가 크게 빨라졌습니다.

### SAC (`soft_actor_critic.py`)
- 조향을 연속 값으로 내는 정책 네트워크와 Q 네트워크 2개를 학습합니다.
- 엔트로피 계수(α)도 함께 학습합니다.
- 타깃 네트워크는 τ 0.01로 천천히 갱신합니다.
- 리플레이 버퍼 20,000, 배치 32를 씁니다.

## 폴더 구성

```
DonkeySimWin/
├── donkey_sim.exe                 # Windows용 DonkeyCar 시뮬레이터
└── projects/mysim/                # 학습 코드
    ├── train_dqn.py
    ├── train_ddqn.py
    ├── train_ddqn_vae.py
    ├── train_sac_vae.py
    ├── soft_actor_critic.py       # SAC, SAC + VAE 에이전트
    ├── vae/vae-level-0-dim-32.pkl # 사전 학습 VAE 가중치
    └── config.py, myconfig.py, manage.py  # donkeycar 프로젝트 설정
```

## 실행

donkeycar와 gym-donkeycar가 설치된 환경에서 실행합니다.

```bash
cd DonkeySimWin/projects/mysim
python train_sac_vae.py --vae ./vae/vae-level-0-dim-32.pkl
```

## 기술 스택

Python, TensorFlow, CUDA, Unity(DonkeyCar Simulator), DQN, DDQN, SAC, VAE
