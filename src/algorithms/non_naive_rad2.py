import torch
from algorithms.non_naive_rad import NonNaiveRAD


class NonNaiveRAD2(NonNaiveRAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(self.apply_aug(_obs), compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()
