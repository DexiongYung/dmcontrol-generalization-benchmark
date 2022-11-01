import utils
import torch
import numpy as np
from copy import deepcopy
from utils import ReplayBuffer
from augmentations import random_shift
from algorithms.sac import SAC
from algorithms.curriculum_learning.curriculum import Curriculum


class AugCL2(Curriculum):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.strong_actor = deepcopy(self.actor)
        self.strong_actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def load_pretrained_agent(self, pretrained_agent: SAC):
        utils.soft_update_params(
            net=pretrained_agent.actor, target_net=self.strong_actor, tau=1
        )
        self.actor = pretrained_agent.actor
        self.actor_optimizer = pretrained_agent.actor_optimizer
        self.critic = pretrained_agent.critic
        self.critic_optimizer = pretrained_agent.critic_optimizer

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            if self.training:
                mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
            else:
                mu, _, _, _ = self.strong_actor(
                    _obs, compute_pi=False, compute_log_pi=False
                )
        return mu.cpu().data.numpy().flatten()

    def update_actor_and_alpha(
        self, obs, obs_aug, L=None, step=None, update_alpha=True
    ):
        mu, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )
            L.log("train_actor/loss", actor_loss, step)
            L.log("train_actor/mean_entropy", entropy.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        strong_mu, pi, log_pi, log_std = self.strong_actor(obs_aug)
        strong_loss = torch.norm(mu.detach() - strong_mu)

        self.strong_actor_optimizer.zero_grad()
        strong_loss.backward()
        self.strong_actor_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = random_shift(obs)
        obs_aug = self.apply_aug(obs)

        self.update_critic(
            obs=obs_shift,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs=obs_shift, obs_aug=obs_aug, L=L, step=step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
