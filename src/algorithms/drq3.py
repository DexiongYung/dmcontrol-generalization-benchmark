import torch
import numpy as np
import augmentations
from algorithms.sac import SAC
import torch.nn.functional as F
from utils import ReplayBuffer


class DrQ3(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.k = args.drq_k
        self.m = 1

    def update_critic(
        self, obs_list, action, reward, next_obs, not_done, L=None, step=None
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(
                augmentations.random_shift(next_obs)
            )
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        Q1_sum = 0
        Q2_sum = 0

        for obs in obs_list:
            curr_Q1, curr_Q2 = self.critic(obs, action)
            Q1_sum += curr_Q1
            Q2_sum += curr_Q2

        critic_loss = F.mse_loss(Q1_sum / self.k, target_Q) + F.mse_loss(
            Q2_sum / self.k, target_Q
        )

        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs_list, L=None, step=None, update_alpha=True):
        actor_loss = 0
        log_pi = None

        for obs in obs_list:
            _, pi, curr_log_pi, log_std = self.actor(obs, detach=True)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss += (self.alpha.detach() * curr_log_pi - actor_Q).mean()

            if log_pi is None:
                log_pi = curr_log_pi
            else:
                log_pi += curr_log_pi

        actor_loss /= self.k
        log_pi /= self.k

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

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_list = [augmentations.random_shift(obs) for _ in range(self.k)]

        self.update_critic(
            obs_list=obs_list,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_list, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
