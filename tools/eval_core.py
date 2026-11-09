from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "on-policy"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.scripts.train.train_mpe import parse_args
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def build_args(seed: int):
    parser = get_config()
    args = parse_args([
        "--env_name", "MPE",
        "--algorithm_name", "rmappo",
        "--experiment_name", "noise_eval",
        "--scenario_name", "simple_spread",
        "--num_agents", "3",
        "--num_landmarks", "3",
        "--episode_length", "25",
        "--seed", str(seed),
        "--use_wandb",
    ], parser)
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    return args


def load_policy(actor_checkpoint: Path, seed: int, device: torch.device):
    args = build_args(seed)
    env = MPEEnv(args)
    policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        env.share_observation_space[0],
        env.action_space[0],
        device=device,
    )
    state = torch.load(actor_checkpoint, map_location=device)
    policy.actor.load_state_dict(state)
    policy.actor.eval()
    return args, env, policy


@torch.no_grad()
def evaluate_episode(actor_checkpoint: Path, episode_seed: int, noise_std: float, device: torch.device):
    args, env, policy = load_policy(actor_checkpoint, episode_seed, device)
    env.seed(episode_seed)
    obs = np.asarray(env.reset(), dtype=np.float32)
    rnn_states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((args.num_agents, 1), dtype=np.float32)
    noise_rng = np.random.default_rng(episode_seed + int(round(noise_std * 10000)) * 100_000)
    physical_dims = obs.shape[-1] - ((args.num_agents - 1) * env.world.dim_c)
    episode_reward = 0.0
    collision_pairs = 0.0
    final_info = None

    for step in range(args.episode_length):
        noisy_obs = obs.copy()
        if noise_std > 0:
            noisy_obs[:, :physical_dims] += noise_rng.normal(
                0.0, noise_std, size=noisy_obs[:, :physical_dims].shape
            ).astype(np.float32)
        actions, rnn_tensor = policy.act(noisy_obs, rnn_states, masks, deterministic=True)
        action_indices = actions.detach().cpu().numpy().astype(int).reshape(args.num_agents)
        rnn_states = rnn_tensor.detach().cpu().numpy()
        actions_env = np.eye(env.action_space[0].n, dtype=np.float32)[action_indices]
        obs_n, reward_n, done_n, info_n = env.step(actions_env)
        obs = np.asarray(obs_n, dtype=np.float32)
        episode_reward += float(reward_n[0][0])
        collision_pairs += sum(float(info["collisions"]) for info in info_n) / 2.0
        final_info = info_n[0]
        masks = np.ones((args.num_agents, 1), dtype=np.float32)
        if all(done_n):
            masks[:] = 0.0

    success = int(final_info["occupied_landmarks"] == args.num_landmarks)
    result = {
        "episode_seed": episode_seed,
        "noise_std": noise_std,
        "episode_reward": episode_reward,
        "success": success,
        "collision_pairs": collision_pairs,
        "final_min_dists": float(final_info["min_dists"]),
        "final_occupied_landmarks": int(final_info["occupied_landmarks"]),
        "episode_length": args.episode_length,
    }
    env.close()
    return result

