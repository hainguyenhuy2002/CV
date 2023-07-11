import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch

def get_scale_from_dataset(dataset):
    scale = None
    if len(dataset) > 0:
        lr = Image.open(dataset[0]['lr'])
        hr = Image.open(dataset[0]['hr'])
        dim1 = round(hr.width / lr.width)
        dim2 = round(hr.height / lr.height)
        scale = max(dim1, dim2)
    return scale


def get_scale(lr, hr):
    dim1 = round(hr.width / lr.width)
    dim2 = round(hr.height / lr.height)
    scale = max(dim1, dim2)
    return scale


def resize_image(lr_image, hr_image, scale=None):
    if scale is None:
        scale = get_scale(lr_image, hr_image)
    if lr_image.width * scale != hr_image.width or lr_image.height * scale != hr_image.height:
        hr_width = lr_image.width * scale
        hr_height = lr_image.height * scale
        return hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)
    return hr_image

def save_checkpoint(epoch, model, optimizer):
    """
    Save model checkpoint.
    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint.pth.tar'
    torch.save(state, filename)
    
def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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