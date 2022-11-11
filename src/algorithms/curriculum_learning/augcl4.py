import numpy as np
import torch
from utils import ReplayBuffer
from augmentations import random_shift
import torch.nn.functional as F
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double


class AugCL4(Curriculum_Double):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.m = 1
        self.k = args.drq_k

    def update_critic(
        self, obs_list, action, reward, next_obs_list, not_done, L=None, step=None
    ):
        target_Q = 0
        target_Q_list = list()
        Q1_list = list()
        Q2_list = list()
        with torch.no_grad():
            for next_obs in next_obs_list:
                _, policy_action, log_pi, _ = self.actor(next_obs)
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
                target_V = (
                    torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
                )
                target_Q += reward + (not_done * self.discount * target_V)
                target_Q_list.append(target_Q.cpu().numpy())

        target_Q /= self.m
        Q1_sum = 0
        Q2_sum = 0

        for obs in obs_list:
            curr_Q1, curr_Q2 = self.critic(obs, action)
            Q1_sum += curr_Q1
            Q2_sum += curr_Q2
            Q1_list.append(curr_Q1.detach().cpu().numpy())
            Q2_list.append(curr_Q2.detach().cpu().numpy())

        critic_loss = F.mse_loss(Q1_sum / self.k, target_Q) + F.mse_loss(
            Q2_sum / self.k, target_Q
        )

        if L is not None:
            Q1_var = (
                0
                if self.k == 1
                else np.concatenate(Q1_list, axis=-1).var(axis=-1).mean()
            )
            Q2_var = (
                0
                if self.k == 1
                else np.concatenate(Q2_list, axis=-1).var(axis=-1).mean()
            )
            Q_target_var = (
                0
                if self.m == 1
                else np.concatenate(target_Q_list, axis=-1).var(axis=-1).mean()
            )

            L.log("train_critic/loss", critic_loss, step)
            L.log(f"train_critic/target_Q_variance_m={self.m}", Q_target_var, step)
            L.log(f"train_critic/Q1_variance_k={self.k}", Q1_var, step)
            L.log(f"train_critic/Q2_variance_k={self.k}", Q2_var, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_list = [self.apply_aug(obs) for _ in range(self.k)]
        next_obs_list = [next_obs]

        self.update_critic(
            obs_list=obs_list,
            action=action,
            reward=reward,
            next_obs_list=next_obs_list,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_list[0], L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
