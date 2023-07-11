from typing import NamedTuple, Tuple, Union

import cv2
import numpy as np
import PIL.Image as pil_image
import torch


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img

def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    img1=img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def compute_metrics(eval_prediction, scale):
    preds = eval_prediction.predictions
    labels = eval_prediction.labels

    # from piq import ssim, psnr
    # print(psnr(denormalize(preds), denormalize(labels), data_range=255.),
    #       ssim(denormalize(preds), denormalize(labels), data_range=255.))

    # original = preds[0][0][0][0]

    preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
    labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

    # print(preds[0][0], original * 255.)

    preds = preds[scale:-scale, scale:-scale]
    labels = labels[scale:-scale, scale:-scale]

    # print(calc_psnr(preds, labels), calc_ssim(preds, labels))

    return {
        'psnr': calc_psnr(preds, labels),
        'ssim': calc_ssim(preds, labels)
    }

def load_image(path):
    return pil_image.open(path).convert('RGB')


def generate_lr(image, scale):
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    return image


def modcrop(image, modulo):
    w = image.width - image.width % modulo
    h = image.height - image.height % modulo
    return image.crop((0, 0, w, h))


def generate_patch(image, patch_size, stride):
    for i in range(0, image.height - patch_size + 1, stride):
        for j in range(0, image.width - patch_size + 1, stride):
            yield image.crop((j, i, j + patch_size, i + patch_size))


def image_to_array(image):
    return np.array(image).transpose((2, 0, 1))

def normalize(x):
    return x / 255.0

def rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.

def load_weights(model, path):
    state_dict = model.state_dict()
    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        labels (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    labels: np.ndarray

class Args:
    def __init__(self, B, U, 
                    num_features, 
                    weights_file, 
                    eval_scale, 
                    lr, batch_size, 
                    num_workers,
                    clip_grad,
                    num_epochs,
                    seed,
                    outputs_dir,
                    eval = True):
        self.B = B
        self.U = U
        self.num_features = num_features
        self.weights_file = weights_file
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.outputs_dir = outputs_dir
        self.clip_grad = clip_grad
        self.num_epochs = num_epochs
        self.seed = seed
        self.eval = eval
        self.eval_scale = eval_scale
