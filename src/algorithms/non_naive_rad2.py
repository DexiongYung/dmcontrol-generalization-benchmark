import torch
from algorithms.rad import RAD


class NonNaiveRAD2(RAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(self.apply_aug(next_obs))
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        aug_obs = self.apply_aug(obs)

        self.update_critic(aug_obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
