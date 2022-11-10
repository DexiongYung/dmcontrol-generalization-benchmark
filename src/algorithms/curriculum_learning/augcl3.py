from utils import ReplayBuffer
from augmentations import random_shift
import torch.nn.functional as F
from algorithms.curriculum_learning.curriculum_double import Curriculum_Double


class AugCL3(Curriculum_Double):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update_critic(
        self, obs_aug, obs_shift, action, reward, next_obs, not_done, L=None, step=None
    ):
        target_Q = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

        current_Q1_shift, current_Q2_shift = self.critic_weak(obs_shift, action)
        critic_loss_shift = F.mse_loss(current_Q1_shift, target_Q) + F.mse_loss(
            current_Q2_shift, target_Q
        )

        current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
        critic_loss_aug = F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(
            current_Q2_aug, target_Q
        )

        if L is not None:
            L.log("train_critic_weak/loss", critic_loss_shift, step)
            L.log("train_critic_strong/loss", critic_loss_aug, step)

        self.critic_weak_optimizer.zero_grad()
        critic_loss_shift.backward()
        self.critic_weak_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss_aug.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs_shift = random_shift(obs)
        obs_aug = self.apply_aug(obs)

        self.update_critic(
            obs_shift=obs_shift,
            obs_aug=obs_aug,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_aug, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
