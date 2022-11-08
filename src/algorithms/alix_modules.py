import math
import torch
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import algorithms.modules as m


def get_local_patches_kernel(kernel_size, device):
    patch_dim = kernel_size**2
    k = th.eye(patch_dim, device=device).view(patch_dim, 1, kernel_size, kernel_size)
    return k


def extract_local_patches(input, kernel, N=None, padding=0, stride=1):
    b, c, _, _ = input.size()
    if kernel is None:
        kernel = get_local_patches_kernel(kernel_size=N, device=input.device)
    flinput = input.flatten(0, 1).unsqueeze(1)
    patches = F.conv2d(flinput, kernel, padding=padding, stride=stride)
    _, _, h, w = patches.size()
    return patches.view(b, c, -1, h, w)


class LearnS(torch.autograd.Function):
    """Uses neighborhood around each feature gradient position to calculate the
    spatial divergence of the gradients, and uses it to update the param S,"""

    @staticmethod
    def forward(ctx, input, param, N, target_capped_ratio, eps):
        """
        input : Tensor
            representation to be processed (used for the gradient analysis).
        param : Tensor
            ALIX parameter S to be optimized.
        N : int
            filter size used to approximate the spatial divergence as a
            convolution (to calculate the ND scores), should be odd, >= 3
        target_capped_ratio : float
            target ND scores used to adaptively tune S
        eps : float
            small stabilization constant for the ND scores
        """
        ctx.save_for_backward(param)
        ctx.N = N
        ctx.target_capped_ratio = target_capped_ratio
        ctx.eps = eps
        return input

    @staticmethod
    def backward(ctx, dy):
        N = ctx.N
        target_capped_ratio = ctx.target_capped_ratio
        eps = ctx.eps
        dy_mean_B = dy.mean(0, keepdim=True)
        ave_dy_abs = th.abs(dy_mean_B)
        pad_Hl = (N - 1) // 2
        pad_Hr = (N - 1) - pad_Hl
        pad_Wl = (N - 1) // 2
        pad_Wr = (N - 1) - pad_Wl
        pad = (pad_Wl, pad_Wr, pad_Hl, pad_Hr)
        padded_ave_dy = F.pad(dy_mean_B, pad, mode="replicate")
        loc_patches_k = get_local_patches_kernel(kernel_size=N, device=dy.device)

        local_patches_dy = extract_local_patches(
            input=padded_ave_dy, kernel=loc_patches_k, stride=1, padding=0
        )
        ave_dy_sq = ave_dy_abs.pow(2)
        patch_normalizer = (N * N) - 1

        unbiased_sq_signal = (
            local_patches_dy.pow(2).sum(dim=2) - ave_dy_sq
        ) / patch_normalizer  # expected squared signal
        unbiased_sq_noise_signal = (local_patches_dy - dy_mean_B.unsqueeze(2)).pow(
            2
        ).sum(
            2
        ) / patch_normalizer  # 1 x C x x H x W expected squared noise

        unbiased_sqn2sig = (unbiased_sq_noise_signal) / (unbiased_sq_signal + eps)

        unbiased_sqn2sig_lp1 = th.log(1 + unbiased_sqn2sig).mean()
        param_grad = target_capped_ratio - unbiased_sqn2sig_lp1

        return dy, param_grad, None, None, None


class Encoder(nn.Module):
    def __init__(self, obs_shape, pretrained=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

        if pretrained:
            pretrained_agent = torch.load(pretrained)
            self.load_state_dict(pretrained_agent.encoder.state_dict())

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")

    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)

    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(
            torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((n_parallel, 1, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        out = torch.matmul(input, self.weight) + self.bias
        if self.n_parallel == 1:
            out = out.squeeze(0)
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class ReprRegularizedEncoder(Encoder):
    """Encoder with regularization applied after final layer."""

    def __init__(self, obs_shape, aug):
        nn.Module.__init__(self)
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.aug = aug
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            self.aug,
        )

        self.apply(weight_init)


class ParameterizedReg(nn.Module):
    """Augmentation/Regularization wrapper where the strength parameterized
    and is tuned with a custom autograd function

     aug : nn.Module
         augmentation/Regularization layer
     parameter_init : float
         initial strength value
     param_grad_fn : str
         custom autograd function to tune the parameter
     param_grad_fn_args : list
         arguments for the custom autograd function
    """

    def __init__(self, aug, parameter_init, param_grad_fn, param_grad_fn_args):
        super().__init__()
        self.aug = aug
        self.P = nn.Parameter(data=torch.tensor(parameter_init))
        self.param_grad_fn_name = param_grad_fn
        if param_grad_fn == "alix_param_grad":
            self.param_grad_fn = LearnS.apply
        else:
            raise NotImplementedError
        self.param_grad_fn_args = param_grad_fn_args

    def forward(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out

    def forward_no_learn(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = self.aug(x, self.P.detach())
        return out

    def forward_no_aug(self, x):
        with torch.no_grad():
            self.P.copy_(torch.clamp(self.P, min=0, max=1))
        out = x
        out = self.param_grad_fn(out, self.P, *self.param_grad_fn_args)
        return out


class NonLearnableParameterizedRegWrapper(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self.aug = aug
        assert isinstance(aug, ParameterizedReg)

    def forward(self, x):
        return self.aug.forward_no_learn(x)


class AllFeatTiedRegularizedEncoder(ReprRegularizedEncoder):
    """Encoder with the same regularization applied after every layer, and with the
    regularization parameter tuned only with the final layer's feature gradients."""

    def __init__(self, obs_shape, aug):
        nn.Module.__init__(self)
        self.aug = aug
        assert len(obs_shape) == 3

        self.repr_dim = 32 * 35 * 35
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            NonLearnableParameterizedRegWrapper(self.aug),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            NonLearnableParameterizedRegWrapper(self.aug),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            NonLearnableParameterizedRegWrapper(self.aug),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            self.aug,
        )
        out = self.convnet(torch.zeros(obs_shape).unsqueeze(0))
        self.out_shape = out.shape

        self.apply(weight_init)


class LocalSignalMixing(nn.Module):
    def __init__(
        self,
        pad,
        fixed_batch=False,
    ):
        """LIX regularization layer

        pad : float
            maximum regularization shift (maximum S)
        fixed batch : bool
            compute independent regularization for each sample (slower)
        """
        super().__init__()
        # +1 to avoid that the sampled values at the borders get smoothed with 0
        self.pad = int(math.ceil(pad)) + 1
        self.base_normalization_ratio = (2 * pad + 1) / (2 * self.pad + 1)
        self.fixed_batch = fixed_batch

    def get_random_shift(self, n, c, h, w, x):
        if self.fixed_batch:
            return torch.rand(size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        else:
            return torch.rand(size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

    def forward(self, x, max_normalized_shift=1.0):
        """
        x : Tensor
            input features
        max_normalized_shift : float
            current regularization shift in relative terms (current S)
        """
        if self.training:
            max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = F.pad(x, padding, "replicate")
            arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
            shift = self.get_random_shift(n, c, h, w, x)
            shift_offset = (1 - max_normalized_shift) / 2
            shift = (shift * max_normalized_shift) + shift_offset
            shift *= (
                2 * self.pad + 1
            )  # can start up to idx 2*pad + 1 - ignoring the left pad
            grid = base_grid + shift
            # normalize in [-1, 1]
            grid = grid * 2.0 / (h + 2 * self.pad) - 1
            return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        else:
            return x

    def get_grid(self, x, max_normalized_shift=1.0):
        max_normalized_shift = max_normalized_shift * self.base_normalization_ratio
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        arange = torch.arange(h, device=x.device, dtype=x.dtype)  # from 0 to eps*h
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 2d grid
        shift = self.get_random_shift(n, c, h, w, x)
        shift_offset = (1 - max_normalized_shift) / 2
        shift = (shift * max_normalized_shift) + shift_offset
        shift *= 2 * self.pad + 1
        grid = base_grid + shift
        # normalize in [-1, 1]
        grid = grid * 2.0 / (h + 2 * self.pad) - 1
        return grid
