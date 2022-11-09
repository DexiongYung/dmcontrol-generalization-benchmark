from algorithms.rad import RAD


class NonNaiveRAD2(RAD):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        aug_obs = self.apply_aug(obs)

        self.update_critic(aug_obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
