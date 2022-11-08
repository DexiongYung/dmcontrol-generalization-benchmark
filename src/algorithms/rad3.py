import torch
from algorithms.rad2 import RAD2


class RAD3(RAD2):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                self.apply_aug(_obs), compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()
