from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "on-policy"))

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.config import get_config
from onpolicy.scripts.train.train_mpe import parse_args


def make_env():
    args = parse_args([
        "--env_name", "MPE", "--algorithm_name", "rmappo",
        "--experiment_name", "test", "--scenario_name", "simple_spread",
        "--num_agents", "3", "--num_landmarks", "3", "--episode_length", "5",
        "--use_wandb"
    ], get_config())
    return MPEEnv(args)


def test_environment_shapes_and_metrics():
    env = make_env()
    obs = np.asarray(env.reset())
    assert obs.shape == (3, 18)
    actions = np.eye(env.action_space[0].n)[[0, 0, 0]]
    next_obs, rewards, dones, infos = env.step(actions)
    assert np.asarray(next_obs).shape == (3, 18)
    assert len(rewards) == 3
    assert {"collisions", "min_dists", "occupied_landmarks"} <= set(infos[0])
    env.close()

