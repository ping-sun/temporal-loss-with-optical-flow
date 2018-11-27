data_info = {
    21: 'VOC',
    2: 'MHP',
}

models_urls = {
    'mbv2_voc': 'https://cloudstor.aarnet.edu.au/plus/s/PsEL9uEuxOtIxJV/download'
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
try:
    from .lw_helpers import maybe_download, batchnorm, conv1x1, conv3x3, convbnrelu, CRPBlock
except ModuleNotFoundError:
    from lw_helpers import maybe_download, batchnorm, conv1x1, conv3x3, convbnrelu, CRPBlock


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Slice0(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, idx):
        return g.op("Slice", x, axes_i=[1], starts_i=[0], ends_i=[idx])

    @staticmethod
    def forward(ctx, x, idx):
        return x.narrow(1, 0, idx)

class Slice1(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, idx):
        return g.op("Slice", x, axes_i=[1], starts_i=[idx], ends_i=[idx + idx])

    @staticmethod
    def forward(ctx, x, idx):
        return x.narrow(1, idx, idx)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2
        self.slice_idx = torch.tensor(0)

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            self.slice_idx = int(x.shape[1] // 2)
            x1 = Slice0.apply(x, self.slice_idx)
            x2 = Slice1.apply(x, self.slice_idx)
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class SNV2(nn.Module):
    def __init__(self, num_classes, input_size=224, width=1.):
        super(SNV2, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = {
            0.5: [-1, 24, 48, 96, 192, 1024],
            1: [-1, 24, 116, 232, 464, 1024],
        }[width]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.layer1 = nn.Sequential(conv_bn(3, input_channel, 2),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        c_layer = 2
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            layers = []
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    layers.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1

        ## Light-Weight RefineNet ##
        for i in range(4, 0, -1):
            setattr(self, f'conv{i}',
                    conv1x1(self.stage_out_channels[i], 256, bias=False)
                    )

        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.segm = conv3x3(256, num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self._initialize_weights()

    def forward(self, x):
                               # channel  size
        l1 = self.layer1(x)    # 24       x / 4
        l2 = self.layer2(l1)   # 48       x / 8
        l3 = self.layer3(l2)   # 96       x / 16
        l4 = self.layer4(l3)   # 192      x / 32

        l4 = self.conv4(l4)
        l4 = self.relu(l4)
        l4 = self.crp4(l4)
        l4 = self.conv_adapt4(l4)
        # l4: x / 32 -> x / 16
        l4 = F.interpolate(l4, scale_factor=2, mode='bilinear', align_corners=False)

        l3 = self.conv3(l3)
        l3 = self.relu(l4 + l3)
        l3 = self.crp3(l3)
        l3 = self.conv_adapt3(l3)
        # l3: x / 16 -> x / 8
        l3 = F.interpolate(l3, scale_factor=2, mode='bilinear', align_corners=False)

        l2 = self.conv2(l2)
        l2 = self.relu(l2 + l3)
        l2 = self.crp2(l2)
        l2 = self.conv_adapt2(l2)
        # l2: x / 8 -> x / 4
        l2 = F.interpolate(l2, scale_factor=2, mode='bilinear', align_corners=False)

        l1 = self.conv1(l1)
        l1 = self.relu(l1 + l2)
        l1 = self.crp1(l1)

        out_segm = self.segm(l1)

        # out: x / 4 -> x
        return F.interpolate(out_segm, scale_factor=4, mode='bilinear', align_corners=False)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)


def snv2(num_classes, pretrained=True, **kwargs):
    """Constructs the network.

    Args:
        num_classes (int): the number of classes for the segmentation head to output.

    """
    model = SNV2(num_classes, **kwargs)
    if pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset == 'MHP':
            bname = 'mbv2_voc'
            key = 'rf_lw' + bname
            url = models_urls[bname]
            state = maybe_download(key, url)
            segm_bias = state.pop('segm.bias')
            state['segm.bias'] = torch.tensor([segm_bias[0], segm_bias[15]])
            segm_weight = state.pop('segm.weight')
            state['segm.weight'] = torch.tensor([segm_weight[0].numpy(), segm_weight[15].numpy()])
            model.load_state_dict(state, strict=False)
            print('model loaded')
    return model

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = snv2(2, False).to(device)
    net.eval()
    x = torch.rand(1, 3, 512, 512, device=device)
    y = net(x)
    print(y.shape)
    # torch.onnx.export(net, x, "temp.onnx", verbose=False)
    from lw_helpers import eval_test
    print('0.5x')
    eval_test(SNV2(2, False, 0.5).to(device))
    print('1x')
    eval_test(SNV2(2, False, 1).to(device))

