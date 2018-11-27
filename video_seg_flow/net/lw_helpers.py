import numpy as np
import torch
import torch.nn as nn

IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os
    import sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes),
                             nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
                             batchnorm(out_planes))


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


def eval_test(net, size=(1, 3, 512, 512)):
    import time
    net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_eval = 100
    # pre run the model
    with torch.no_grad():
        for i in range(10):
            net(torch.rand(size, device=device))
    # test without GPU IO
    t = time.time()
    with torch.no_grad():
        for i in range(n_eval):
            net(torch.rand(size, device=device))
    torch.cuda.synchronize()
    print(net.__class__.__name__, 'Without IO', f'{(time.time() - t)/n_eval:.4f} s')
    # test consider GPU IO
    t = time.time()
    with torch.no_grad():
        for i in range(n_eval):
            x = torch.rand(size)
            x = x.to(device, non_blocking=True)
            y = net(x)
            y.to('cpu')
    torch.cuda.synchronize()
    print(net.__class__.__name__, 'With    IO', f'{(time.time() - t)/n_eval:.4f} s')
