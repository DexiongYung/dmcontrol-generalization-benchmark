import torch
import numpy as np
from algorithms.sac import SAC
import algorithms.modules as m
import algorithms.alix_modules as am
import torch.nn.functional as F
from copy import deepcopy


def make_optimizer(encoder, encoder_lr, **kwargs):
    encoder_params = list(encoder.parameters())
    encoder_aug_parameters = list(encoder.aug.parameters())
    encoder_non_aug_parameters = [
        p
        for p in encoder_params
        if all([p is not aug_p for aug_p in encoder_aug_parameters])
    ]
    return torch.optim.Adam(
        [
            {"params": encoder_non_aug_parameters},
            {"params": encoder_aug_parameters, **kwargs},
        ],
        lr=encoder_lr,
    )


class ALIX(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        shared_cnn = am.AllFeatTiedRegularizedEncoder(
            obs_shape=obs_shape,
            aug=am.ParameterizedReg(
                aug=am.LocalSignalMixing(pad=2, fixed_batch=True),
                parameter_init=0.5,
                param_grad_fn="alix_param_grad",
                param_grad_fn_args=[3, 0.535, 1e-20],
            ),
        )
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, args.num_head_layers, args.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )

        self.actor = m.Actor(
            actor_encoder,
            action_shape,
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.encoder_optimizer = make_optimizer(
            encoder=shared_cnn, encoder_lr=1e-4, betas=[0.5, 0.999], lr=2e-3
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        target_Q = self.calculate_target_Q(
            next_obs=next_obs, reward=reward, not_done=not_done
        )

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
