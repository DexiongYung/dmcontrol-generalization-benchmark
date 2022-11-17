import torch
import numpy as np
import augmentations
from algorithms.sac import SAC
import torch.nn.functional as F
from utils import ReplayBuffer


class Curriculum_Anneal(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.k = args.drq_k
        self.train_steps = args.train_steps
        self.switch_step = args.switch_step

    def update_critic(
        self, obs_list, action, reward, next_obs, not_done, L=None, step=None
    ):
        target_Q = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

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

    def weight_aug_obs(self, obs, step):
        obs_list = list()
        for _ in range(self.k):
            obs_aug = augmentations.random_shift(obs)
            if step > self.switch_step:
                aug_weight = (step - self.switch_step) / self.train_steps
                obs_aug = (1 - aug_weight) * obs_aug + aug_weight * self.apply_aug(
                    obs_aug
                )
            obs_list.append(obs_aug)
        return obs_list

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_list = self.weight_aug_obs(obs=obs, step=step)

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
