import torch
from utils import ReplayBuffer
from algorithms.sac import SAC
from algorithms.curriculum_learning.curriculum import Curriculum


class SECANT(Curriculum):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def load_pretrained_agent(self, pretrained_agent: SAC):
        self.expert = pretrained_agent.actor

    def update(self, replay_buffer: ReplayBuffer, L, step):
        obs, _, _, _, _ = replay_buffer.sample()
        obs_aug = self.apply_aug(obs)

        mu, _, _, _ = self.actor(obs_aug)
        expert_mu, _, _, _ = self.expert(obs)

        loss = torch.norm(mu - expert_mu)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
