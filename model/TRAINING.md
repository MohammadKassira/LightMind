# Running DQN Training

## Prerequisites

1. SUMO installed and `SUMO_HOME` set in your environment.
2. Python packages: `pip install torch pyyaml`

## How to run

Open `run_training.py` and edit the four lines at the top of the `── CONFIG ──` block:

```python
NET_FILE   = "networks/external/RESCO/cologne1/cologne1.net.xml"
ROUTE_FILE = "networks/external/RESCO/cologne1/cologne1.rou.xml"
BEGIN_TIME = 25200
RUN_NAME   = "cologne1_dqn"
```

Then run:

```bash
python run_training.py
```

All other hyperparameters (episodes, learning rate, epsilon schedule, etc.) are in `configs/train.yaml`.

## Outputs

```
checkpoints/<RUN_NAME>/final.pt        # model weights + optimizer state
checkpoints/<RUN_NAME>/metrics.json    # episode_returns, losses, q_mean, etc.
checkpoints/<RUN_NAME>/ep00100.pt      # periodic checkpoints (every 100 episodes)
```

Load `metrics.json` with any JSON reader for plotting.
