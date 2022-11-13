import torch
import augmentations
from algorithms.svea import SVEA


class SVEA2(SVEA):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(
                augmentations.random_shift(_obs), compute_log_pi=False
            )
        return pi.cpu().data.numpy().flatten()
