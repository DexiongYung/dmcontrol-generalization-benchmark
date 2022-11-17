import torch
import numpy as np
import augmentations
import torch.nn.functional as F
from logger import Logger
import utils
from algorithms.rad import RAD


class SVEA2(RAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.svea_alpha = args.svea_alpha
        self.svea_beta = args.svea_beta

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(
                augmentations.random_shift(next_obs)
            )
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # if self.svea_alpha == self.svea_beta:
        #     obs = utils.cat(obs, self.aug_func(obs.clone()))
        #     action = utils.cat(action, action)
        #     target_Q = utils.cat(target_Q, target_Q)

        #     current_Q1, current_Q2 = self.critic(obs, action)
        #     critic_loss = (self.svea_alpha + self.svea_beta) * (
        #         F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #     )
        # else:
        #     current_Q1, current_Q2 = self.critic(obs, action)
        #     critic_loss = self.svea_alpha * (
        #         F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        #     )

        #     obs_aug = self.aug_func(obs.clone())
        #     current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        #     critic_loss += self.svea_beta * (
        #         F.mse_loss(current_Q1_aug, target_Q)
        #         + F.mse_loss(current_Q2_aug, target_Q)
        #     )

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss_shift = self.svea_alpha * (
            F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        )

        obs_aug = self.apply_aug(obs.clone())
        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        critic_loss_aug = self.svea_beta * (
            F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q)
        )

        critic_loss = critic_loss_shift + critic_loss_aug

        if L is not None:
            L.log("train_critic/shift_loss", critic_loss_shift, step)
            L.log("train_critic/aug_loss", critic_loss_aug, step)
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        _, pi, strong_log_pi, log_std = self.actor(self.apply_aug(obs), detach=True)
        actor_loss += (self.alpha.detach() * strong_log_pi - actor_Q).mean()
        actor_loss /= 2

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

    def update(self, replay_buffer: utils.ReplayBuffer, L: Logger, step: int):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = augmentations.random_shift(obs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
