import torch
import numpy as np
import augmentations
from algorithms.sac import SAC
import torch.nn.functional as F
from utils import ReplayBuffer


class DrQ2(SAC):  # [K=1, M=1]
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.k = args.drq_k
        self.m = args.drq_m

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
        critic_loss = 0

        for obs in obs_list:
            curr_Q1, curr_Q2 = self.critic(obs, action)
            critic_loss += F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)
            Q1_list.append(curr_Q1.detach().cpu().numpy())
            Q2_list.append(curr_Q2.detach().cpu().numpy())

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

    def update_actor_and_alpha(self, obs_list, L=None, step=None, update_alpha=True):
        actor_loss = 0
        entropy = list()
        for obs in obs_list:
            _, pi, log_pi, log_std = self.actor(obs, detach=True)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss += (self.alpha.detach() * log_pi - actor_Q).mean()
            entropy.append(
                (
                    0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi))
                    + log_std.sum(dim=-1)
                ).mean().detach().cpu().numpy()
            )

        if L is not None:
            entropy = np.array(entropy)
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
        obs_list = [augmentations.random_shift(obs, 4) for _ in range(self.k)]
        next_obs_list = [augmentations.random_shift(next_obs, 4) for _ in range(self.m)]

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
            self.update_actor_and_alpha(obs_list, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
