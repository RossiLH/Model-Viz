from __future__ import annotations
import torch


def flow_matching_scheduler(num_steps: int = 10) -> tuple:
    timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps)
    dt = -1.0 / num_steps
    return timesteps, dt


def sample_noise(batch_size: int, action_horizon: int, action_dim: int) -> torch.Tensor:
    return torch.randn(batch_size, action_horizon, action_dim)


def denoise_step(model_fn, x_t, timestep, prefix_kv, dt: float):
    velocity = model_fn(x_t, timestep, prefix_kv)
    return x_t + dt * velocity