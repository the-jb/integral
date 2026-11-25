"""Sampling of the flow-matching image generation: the timestep schedules, classifier-free guidance and the solvers
(Euler, second-order multistep DPM-Solver++). `t = 1` is noise and `t = 0` the clean latent; the model predicts the
velocity `noise - latent`."""

from collections.abc import Callable

import torch
from torch import Tensor


def flow_timesteps(num_inference_steps: int, time_shift: float, device=None, dtype=torch.float32) -> Tensor:
    """
    The timesteps of a sampling run, from 1 to 0.

    Args:
        num_inference_steps (`int`):
            Number of solver steps.
        time_shift (`float`):
            The schedule shift `s`: `t = s * u / (1 + (s - 1) * u)` over `u` uniform in [1, 0]. 1.0 leaves `u` as is.

    Returns:
        `torch.FloatTensor` of shape `(num_inference_steps + 1,)`, descending.
    """
    u = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=dtype)
    return u if time_shift == 1.0 else time_shift * u / (1 + (time_shift - 1) * u)


def classifier_free_guidance(velocity: Tensor, unconditional_velocity: Tensor, guidance_scale: float) -> Tensor:
    """`unconditional + guidance_scale * (conditional - unconditional)`, in fp32."""
    unconditional_velocity = unconditional_velocity.float()
    return unconditional_velocity + guidance_scale * (velocity.float() - unconditional_velocity)


def euler_step(latents: Tensor, velocity: Tensor, t: Tensor, t_next: Tensor) -> Tensor:
    """One Euler step of the flow ODE from `t` to `t_next`: `latents + (t_next - t) * velocity`."""
    return latents + (t_next - t) * velocity


def dpm_solver_timesteps(num_inference_steps: int, time_shift: float, device=None) -> Tensor:
    """
    The timesteps of a DPM-Solver++ run, from `1 - 0.001` shifted down to 0.

    Args:
        num_inference_steps (`int`):
            Number of solver steps.
        time_shift (`float`):
            The schedule shift `s`: `t = s * u / (1 + (s - 1) * u)` over `u` uniform in [0.999, 0].

    Returns:
        `torch.FloatTensor` of shape `(num_inference_steps + 1,)`, descending.
    """
    u = 1.0 - torch.linspace(1.0, 0.001, num_inference_steps + 1).to(device)
    return (time_shift * u / (1 + (time_shift - 1) * u)).flip(dims=[0])


def _half_log_snr(t: Tensor) -> Tensor:
    """`log(alpha_t) - log(sigma_t)` with `alpha_t = 1 - t`, `sigma_t = t`."""
    return torch.log(1 - t) - torch.log(t)


def _dpm_solver_first_order_update(latents: Tensor, data: Tensor, s: Tensor, t: Tensor) -> Tensor:
    """First-order DPM-Solver++ update from `s` to `t` with the data prediction at `s`."""
    h = _half_log_snr(t) - _half_log_snr(s)
    alpha_t = torch.exp(torch.log(1 - t))
    phi_1 = torch.expm1(-h)
    return t / s * latents - alpha_t * phi_1 * data


def _dpm_solver_second_order_update(
    latents: Tensor, data_prev: Tensor, data: Tensor, s_prev: Tensor, s: Tensor, t: Tensor
) -> Tensor:
    """Second-order multistep DPM-Solver++ update from `s` to `t` with the data predictions at `s_prev` and `s`."""
    lambda_prev, lambda_s, lambda_t = _half_log_snr(s_prev), _half_log_snr(s), _half_log_snr(t)
    alpha_t = torch.exp(torch.log(1 - t))
    h_prev = lambda_s - lambda_prev
    h = lambda_t - lambda_s
    r = h_prev / h
    d = (1.0 / r) * (data - data_prev)
    phi_1 = torch.expm1(-h)
    return (t / s) * latents - (alpha_t * phi_1) * data - 0.5 * (alpha_t * phi_1) * d


def dpm_solver_sample(velocity_fn: Callable[[Tensor, Tensor], Tensor], latents: Tensor, timesteps: Tensor) -> Tensor:
    """
    Integrates the flow ODE with the second-order multistep DPM-Solver++ (data prediction), one model call per step.
    The first and the last step are first order.

    Args:
        velocity_fn (`Callable`):
            `velocity_fn(latents, t)` -> the velocity at `t` (a 0-dim tensor). It is called at `t * 1000 / 1000`, the
            timestep taken to the solver's 1000-step time scale and back.
        latents (`torch.FloatTensor`):
            The noise at `timesteps[0]`.
        timesteps (`torch.FloatTensor` of shape `(num_inference_steps + 1,)`):
            `dpm_solver_timesteps`; at least two steps.

    Returns:
        `torch.FloatTensor`: the latents at `timesteps[-1]`.
    """
    num_steps = timesteps.shape[0] - 1
    if num_steps < 2:
        raise ValueError(f"DPM-Solver++ needs at least 2 steps, got {num_steps}")
    data_prev = None
    for i in range(num_steps):
        s, t = timesteps[i], timesteps[i + 1]
        velocity = velocity_fn(latents, s * 1000 / 1000).float()
        noise = (1 - s) * velocity + latents
        data = (latents - s * noise) / (1 - s)
        if i == 0 or i == num_steps - 1:
            latents = _dpm_solver_first_order_update(latents, data, s, t)
        else:
            latents = _dpm_solver_second_order_update(latents, data_prev, data, timesteps[i - 1], s, t)
        data_prev = data
    return latents
