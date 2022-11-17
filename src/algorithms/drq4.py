import torch
import numpy as np
import augmentations
from algorithms.sac import SAC
import torch.nn.functional as F
from utils import ReplayBuffer


class DrQ4(SAC):
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
            self.update_actor_and_alpha(obs_list[0], L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
