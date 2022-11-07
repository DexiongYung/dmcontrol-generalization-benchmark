import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os


dataloader = None
data_iter = None
dataloader_batch_sz = None


def get_jitter_model(x):
    return TF.ColorJitter(0, (0, 2), (0, 3), (-0.5, 0.5)).to(x.get_device())


def _load_data(
    sub_path: str, batch_size: int = 256, image_size: int = 84, num_workers: int = 16
):
    global data_iter, dataloader
    for data_dir in utils.load_config("datasets"):
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, sub_path)
            if not os.path.exists(fp):
                print(f"Warning: path {fp} does not exist, falling back to {data_dir}")
            dataloader = torch.utils.data.DataLoader(
                datasets.ImageFolder(
                    fp,
                    TF.Compose(
                        [
                            TF.RandomResizedCrop(image_size),
                            TF.RandomHorizontalFlip(),
                            TF.ToTensor(),
                        ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            data_iter = iter(dataloader)
            break
    if data_iter is None:
        raise FileNotFoundError(
            "failed to find image data at any of the specified paths"
        )
    print("Loaded dataset from", data_dir)


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
    partition = "val" if use_val else "train"
    sub_path = os.path.join("places365_standard", partition)
    print(f"Loading {partition} partition of places365_standard...")
    _load_data(
        sub_path=sub_path,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )


def _load_coco(batch_size=256, image_size=84, num_workers=16, use_val=False):
    sub_path = "COCO"
    print(f"Loading COCO 2017 Val...")
    _load_data(
        sub_path=sub_path,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )


def _get_data_batch(batch_size):
    global data_iter
    try:
        imgs, _ = next(data_iter)

        if batch_size < imgs.size(0):
            imgs = imgs[:batch_size]
        if imgs.size(0) < batch_size:
            data_iter = iter(dataloader)
            imgs, _ = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        imgs, _ = next(data_iter)
    return imgs.cuda()


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3

    imgs = imgs.view([b, frames, 3, h, w])
    imgs = (
        imgs[:, :, 0, ...] * 0.2989
        + imgs[:, :, 1, ...] * 0.587
        + imgs[:, :, 2, ...] * 0.114
    )

    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(
        device
    )  # broadcast tiling
    return imgs


def random_grayscale(images, p=0.3):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: cpu or cuda
    returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.0
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.0
    return out


def random_flip(images, p=0.2):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: cpu or gpu,
    p: prob of applying aug,
    returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape

    images = images.to(device)

    flipped_images = images.flip([3])

    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1]  # // 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)

    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]

    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out


def random_rotation(images, p=0.95):
    """
    args:
    imgs: torch.tensor shape (B,C,H,W)
    device: str, cpu or gpu,
    p: float, prob of applying aug,
    returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape

    images = images.to(device)

    rot90_images = images.rot90(1, [2, 3])
    rot180_images = images.rot90(2, [2, 3])
    rot270_images = images.rot90(3, [2, 3])

    rnd = np.random.uniform(0.0, 1.0, size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)

    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[torch.where(mask == i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(
            device
        )
        m = m[:, :, None, None]
        masks[i] = m

    out = (
        masks[0] * images
        + masks[1] * rot90_images
        + masks[2] * rot180_images
        + masks[3] * rot270_images
    )

    out = out.view([bs, -1, h, w])
    return out


def pad_to_shape(arr, out_shape):
    c, h, w = arr.shape
    m, n = out_shape
    out = torch.zeros(c, m, n)
    mx, my = (m - h) // 2, (n - w) // 2
    out[:, mx : mx + h, my : my + w] = arr
    return out


def color_jitter(imgs, p=1):
    """
    inputs np array outputs tensor
    """
    b, c, h, w = imgs.shape
    num_frames = int(c / 3)
    num_samples = int(p * b * num_frames)

    sampled_idxs = torch.from_numpy(np.random.randint(0, b, num_samples))
    imgs = imgs.view(-1, 3, h, w)
    model = get_jitter_model(x=imgs)
    imgs[sampled_idxs] = model(imgs[sampled_idxs])
    return imgs.view(b, c, h, w)


def random_perspective(imgs, p=1):
    """
    inputs np array outputs tensor
    """
    b, c, h, w = imgs.shape
    imgs = imgs.view(-1, 3, h, w) / 255.0
    model = kornia.augmentation.RandomPerspective(p=p)
    imgs = model(imgs)
    return imgs.view(b, c, h, w) * 255.0


def random_resize_crop(imgs, p=1):
    b, c, h, w = imgs.shape
    imgs = imgs.view(-1, 3, h, w)
    model = TF.RandomResizedCrop(size=h, scale=(0.2, 1))
    imgs = model(imgs)
    return imgs.view(b, c, h, w)


def load_dataloader(batch_size, image_size, dataset="coco"):
    global dataloader
    global dataloader_batch_sz

    if dataloader_batch_sz is not None and batch_size > dataloader_batch_sz:
        dataloader = None

    if dataloader is None:
        dataloader_batch_sz = batch_size
        if dataset == "places365_standard":
            _load_places(batch_size=batch_size, image_size=image_size)
        elif dataset == "coco":
            _load_coco(batch_size=batch_size, image_size=image_size)
        else:
            raise NotImplementedError(
                f'overlay has not been implemented for dataset "{dataset}"'
            )


def random_overlay(x, dataset="coco"):
    """Randomly overlay an image from Places or COCO"""
    global data_iter
    alpha = 0.5

    load_dataloader(batch_size=x.size(0), image_size=x.size(-1), dataset=dataset)

    imgs = _get_data_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)

    return ((1 - alpha) * (x / 255.0) + (alpha) * imgs) * 255.0


def random_np_overlay(x, dataset="coco"):
    """Randomly overlay an image from Places or COCO"""
    global data_iter
    alpha = 0.5

    load_dataloader(batch_size=x.size(0), image_size=x.size(-1), dataset=dataset)

    imgs = _get_data_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)

    return torch.from_numpy(
        (
            (1 - alpha) * (x.cpu().detach().numpy() / 255.0)
            + (alpha) * imgs.cpu().data.numpy()
        )
        * 255.0
    )


def mix_up(x, dataset="coco"):
    global data_iter

    load_dataloader(batch_size=x.size(0), image_size=x.size(-1), dataset=dataset)

    imgs = _get_data_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    weights = (
        torch.from_numpy(np.random.uniform(size=x.shape[0]))
        .to(x.get_device())
        .view(-1, 1, 1, 1)
    )

    return ((1 - weights) * (x / 255.0) + (weights) * imgs) * 255.0


def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)


def create_hsv_mask(x_rgb, hue_thres, sat_thres, val_thres):
    x_hsv = kornia.color.rgb_to_hsv(x_rgb)
    b, _, h, w = x_hsv.shape
    weight = torch.FloatTensor([hue_thres, sat_thres, val_thres]).to(x_rgb.get_device())
    weight = weight.view(1, -1, 1, 1).repeat(b, 1, h, w)
    mask = x_hsv > weight.to(x_rgb.get_device())
    mask = torch.all(mask, dim=1)
    return mask.unsqueeze(1).repeat(1, x_rgb.shape[1], 1, 1)


def create_rgb_mask(x_rgb, R_thres, G_thres, B_thres):
    weight = torch.ones(x_rgb.shape)
    weight[:, 0] = weight[:, 0] * R_thres
    weight[:, 1] = weight[:, 1] * G_thres
    weight[:, 2] = weight[:, 2] * B_thres
    return x_rgb > weight.to(x_rgb.get_device())


def splice(x, hue_thres=0, sat_thres=0, val_thres=0.6):
    # 0.6 val for video_hard, 0.4 for video_easy
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    x_rgb[~mask] = overlay[~mask]
    return x_rgb.reshape(n, c, h, w) * 255.0


def DrQ2_random_shift(x, pad=4):
    n, _, h, w = x.size()
    assert h == w
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype
    )[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

    shift = torch.randint(
        0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
    )
    shift *= 2.0 / (h + 2 * pad)

    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def splice_mix_up(x, hue_thres=0, sat_thres=0, val_thres=0.6):
    # 0.6 val for video_hard, 0.4 for video_easy
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    x_rgb[~mask] = 0.5 * x_rgb[~mask] + 0.5 * overlay[~mask]
    return x_rgb.reshape(n, c, h, w) * 255.0


def splice_mix_up_jitter(x, hue_thres=3.5, sat_thres=0, val_thres=0):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    model = get_jitter_model(x=x)
    for i in range(n):
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        out = model(temp_x)
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    total_out[mask] = overlay[mask] * 0.5 + x_rgb[mask] * 0.5
    return total_out.reshape(n, c, h, w) * 255.0


def CS_splice(x, hue_thres=3.5, sat_thres=0, val_thres=0):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    mask2 = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=0, sat_thres=sat_thres, val_thres=0.6
    )
    overlay[mask2] = x_rgb[mask2]
    overlay[~mask] = x_rgb[~mask]
    return overlay.reshape(n, c, h, w) * 255.0


def splice_conv(x, hue_thres=3.5, sat_thres=0, val_thres=0):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    mask2 = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=0, sat_thres=sat_thres, val_thres=0.6
    )
    conv = random_conv(x=x)
    conv = conv.reshape(-1, 3, h, w) / 255.0
    overlay[mask2] = conv[mask2]
    overlay[~mask] = conv[~mask]
    return overlay.reshape(n, c, h, w) * 255.0


def emphasize(x, hue_thres=3.5, sat_thres=0, val_thres=0):
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    mask2 = create_hsv_mask(
        x_rgb=x_rgb, hue_thres=0, sat_thres=sat_thres, val_thres=0.6
    )
    x_rgb[mask2] *= 2
    x_rgb[~mask] *= 2
    return x_rgb.reshape(n, c, h, w) * 255.0


def CS_splice_2x_jitter(x, hue_thres=3.5, sat_thres=0, val_thres=0):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    model = get_jitter_model(x=x)
    for i in range(n):
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        mask = create_hsv_mask(
            x_rgb=temp_x, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
        )
        mask2 = create_hsv_mask(
            x_rgb=temp_x, hue_thres=0, sat_thres=sat_thres, val_thres=0.6
        )
        out = model(temp_x)
        out_2 = model(temp_x)
        out[mask2] = out_2[mask2]
        out[~mask] = out_2[~mask]
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w) * 255.0


def splice_jitter(x, hue_thres=0, sat_thres=0, val_thres=0.55):
    global data_iter
    load_dataloader(batch_size=x.size(0), image_size=x.size(-1))
    overlay = _get_data_batch(x.size(0)).repeat(x.size(1) // 3, 1, 1, 1)
    n, c, h, w = x.shape
    x_rgb = x.reshape(-1, 3, h, w) / 255.0
    mask = create_hsv_mask(
        x_rgb, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
    )
    model = get_jitter_model(x=x)
    for i in range(n):
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        out = model(temp_x)
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    total_out[~mask] = overlay[~mask]
    return total_out.reshape(n, c, h, w) * 255.0


def splice_2x_conv(x, hue_thres=0, sat_thres=0, val_thres=0.6):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        mask = create_hsv_mask(
            temp_x, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
        )
        temp_x = F.pad(temp_x, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        out2 = torch.sigmoid(F.conv2d(temp_x, weights)) * 255.0
        out[mask] = out2[mask]
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)


def splice_2x_jitter(x, hue_thres=0, sat_thres=0, val_thres=0.6):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    model = get_jitter_model(x=x)
    for i in range(n):
        temp_x = x[i : i + 1].reshape(-1, 3, h, w) / 255.0
        mask = create_hsv_mask(
            temp_x, hue_thres=hue_thres, sat_thres=sat_thres, val_thres=val_thres
        )
        out = model(temp_x)
        out_2 = model(temp_x)
        out[~mask] = out_2[~mask]
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w) * 255.0


def random_cutout_color(imgs, min_cut=10, max_cut=30):
    """
    args:
    imgs: shape (B,C,H,W)
    out: output size (e.g. 84)
    """
    imgs = imgs.detach().cpu().numpy()
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.0
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()

        # add random box
        cut_img[:, h11 : h11 + h11, w11 : w11 + w11] = np.tile(
            rand_box[i].reshape(-1, 1, 1),
            (1,) + cut_img[:, h11 : h11 + h11, w11 : w11 + w11].shape[1:],
        )

        cutouts[i] = cut_img
    return torch.from_numpy(cutouts)


def batch_from_obs(obs, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
    """Prepare batch for self-supervised policy adaptation at test-time"""
    batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
    batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
    batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

    return random_crop(batch_obs), random_crop(batch_next_obs), batch_action


def identity(x):
    return x


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"
    assert isinstance(x, torch.Tensor) and x.is_cuda, "input must be CUDA tensor"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def DrAC_crop(x, size: int = 84, pad: int = 16):
    return torch.nn.Sequential(
        torch.nn.ReplicationPad2d(pad), kornia.augmentation.RandomCrop((size, size))
    )(x)


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


aug_to_func = {
    "grayscale": random_grayscale,
    "flip": random_flip,
    "rotate": random_rotation,
    "conv": random_conv,
    "shift": random_shift,
    "color_jitter": color_jitter,
    "splice_mix_up": splice_mix_up,
    "identity": identity,
    "splice": splice,
    "crop": random_crop,
    "drac_crop": DrAC_crop,
    "cutout_color": random_cutout_color,
    "splice_jitter": splice_jitter,
    "CS_splice": CS_splice,
    "overlay": random_overlay,
    "splice_2x_conv": splice_2x_conv,
    "splice_2x_jitter": splice_2x_jitter,
    "splice_mix_up_jitter": splice_mix_up_jitter,
    "random_perspective": random_perspective,
    "random_resize_crop": random_resize_crop,
    "DrQ2_random_shift": DrQ2_random_shift,
    "mix_up": mix_up,
    "overlay_np": random_np_overlay,
    "emphasize": emphasize,
    "splice_conv": splice_conv,
}
