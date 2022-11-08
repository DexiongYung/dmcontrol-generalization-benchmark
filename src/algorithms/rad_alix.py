import json
import torch
import augmentations
from algorithms.alix import ALIX


class RAD_ALIX(ALIX):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        aug_keys = args.data_aug.split("-")
        aug_params = json.loads(args.aug_params) if args.aug_params else {}
        self.aug_funcs = dict()
        for key in aug_keys:
            self.aug_funcs[key] = dict(
                func=augmentations.aug_to_func[key], params=aug_params.get(key, {})
            )

    def apply_aug(self, x):
        for _, aug_dict in self.aug_funcs.items():
            x = aug_dict["func"](x, **aug_dict["params"])

        return x

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                self.apply_aug(_obs), compute_pi=False, compute_log_pi=False
            )
        return mu.cpu().data.numpy().flatten()
