from utils import ReplayBuffer
from algorithms.curriculum_learning.curriculum import Curriculum

class CurriculumSingle(Curriculum):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = self.apply_aug(obs)

        self.update_critic(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            L=L,
            step=step,
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
