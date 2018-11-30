"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

try:
    from .lw_helpers import maybe_download, batchnorm, conv1x1, conv3x3, convbnrelu, CRPBlock
except ModuleNotFoundError:
    from lw_helpers import maybe_download, batchnorm, conv1x1, conv3x3, convbnrelu, CRPBlock


data_info = {
    21: 'VOC',
    2: 'MHP',
}

models_urls = {
    'mbv2_voc': 'https://cloudstor.aarnet.edu.au/plus/s/PsEL9uEuxOtIxJV/download'
}

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block from https://arxiv.org/abs/1801.04381"""
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return (out + residual)
        else:
            return out

# tensor for global pooling vectors' multiplication
class tensor_multiply_block(nn.Module):
    def __init__(self,in_planes):
        super(tensor_multiply_block,self).__init__()
        self.tensor = nn.Parameter(
            torch.Tensor(1, in_planes, 1, 1).normal_(0, 0.01)
        )

    def forward(self, x):
        out = self.tensor * x
        return out



class MBv2(nn.Module):
    """Net Definition"""
    mobilenet_config = [
        [1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    layer_num = [[3,24],[4,32],[5,64],[6,96],[7,160],[8,320]]
    global_channel = 3
    in_planes = 32 # number of input channels
    num_layers = len(mobilenet_config)
    def __init__(self, num_classes):
        super(MBv2, self).__init__()

        self.layer1 = convbnrelu(3, self.in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (self.mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_planes, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_planes = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1
        # initialize tensor
        for num_layer,c in self.layer_num:
            for i in range(1,self.global_channel+1):
                setattr(self,'layer{}'.format(str(num_layer)+'_'+str(i)),tensor_multiply_block(c))


        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
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

    def feature(self, x):
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        return l3, l4, l5, l6, l7, l8

    def global_feature(self, x, mask, feature=None):
        if feature is None:
            l3, l4, l5, l6, l7, l8 = self.feature(x)
        else:
            l3, l4, l5, l6, l7, l8 = feature
        gl_l3 = self.get_global_feature(l3, mask)
        gl_l4 = self.get_global_feature(l4, mask)
        gl_l5 = self.get_global_feature(l5, mask)
        gl_l6 = self.get_global_feature(l6, mask)
        gl_l7 = self.get_global_feature(l7, mask)
        gl_l8 = self.get_global_feature(l8, mask)
        gl_l3_bg = self.get_global_feature(l3, 1 - mask)
        gl_l4_bg = self.get_global_feature(l4, 1 - mask)
        gl_l5_bg = self.get_global_feature(l5, 1 - mask)
        gl_l6_bg = self.get_global_feature(l6, 1 - mask)
        gl_l7_bg = self.get_global_feature(l7, 1 - mask)
        gl_l8_bg = self.get_global_feature(l8, 1 - mask)
        return [gl_l3, gl_l4, gl_l5, gl_l6, gl_l7, gl_l8, gl_l3_bg, \
                  gl_l4_bg, gl_l5_bg, gl_l6_bg, gl_l7_bg, gl_l8_bg], feature

    def forward_all(self, x, global_feature=None, feature=None):

        if feature is None:
            with torch.no_grad():
                feature = self.feature(x)
        l3, l4, l5, l6, l7, l8 = feature

        if global_feature is not None:
            gl_l3, gl_l4, gl_l5, gl_l6, gl_l7, gl_l8, gl_l3_bg, \
            gl_l4_bg, gl_l5_bg, gl_l6_bg, gl_l7_bg, gl_l8_bg = global_feature  # get last frame's features
            l8 = self.layer8_1(l8) + self.layer8_3(gl_l8_bg) #  tensor multiplication and summation
            l7 = self.layer7_1(l7) + self.layer7_3(gl_l7_bg)
            l6 = self.layer6_1(l6) + self.layer6_3(gl_l6_bg)
            l5 = self.layer5_1(l5) + self.layer5_3(gl_l5_bg)
            l4 = self.layer4_1(l4) + self.layer4_3(gl_l4_bg)
            l3 = self.layer3_1(l3) + self.layer3_3(gl_l3_bg)
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, scale_factor=2, mode='bilinear', align_corners=False)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, scale_factor=2, mode='bilinear', align_corners=False)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, scale_factor=2, mode='bilinear', align_corners=False)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        out_segm = self.segm(l3)
        out_segm = F.interpolate(out_segm, scale_factor=4, mode='bilinear', align_corners=False)

        return out_segm, feature

    def forward(self, x, global_feature):
        return self.forward_all(x, global_feature)[0]

    def prediction(self, x, global_feature):
        out_segm, feature = self.forward_all(x, global_feature)
        n, c, h, w = x.shape
        mask = (out_segm.argmax(dim=1)==1).float().reshape(n, 1, h, w)
        global_feature, _ = self.global_feature(x, mask, feature)
        return out_segm, global_feature

    def prediction_soft(self, x, global_feature):
        out_segm, feature = self.forward_all(x, global_feature)
        n, c, h, w = x.shape
        mask = F.softmax(out_segm, 1)[:, 1].unsqueeze(1)
        global_feature, _ = self.global_feature(x, mask, feature)
        return out_segm, global_feature, feature



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # global pooling with mask
    def get_global_feature(self, features, mask):
        # mask = F.avg_pool2d(mask, (mask.size(2)//features.size(2), mask.size(3)//features.size(3)))
        h, w = features.size()[2:]
        # kernel_size = w // 8 * 2 + 1
        # padding = w // 8
        # stride = min(h, w) // 16
        mask = F.adaptive_avg_pool2d(mask, features.size()[2:])
        mask = mask.unsqueeze(1)
        features = features * mask
        # features = F.avg_pool2d(features,features.size()[2:])
        # features = F.avg_pool2d(features, (features.size(2)//2, features.size(3)//2))
        # features = F.avg_pool2d(features, 2)
        # mask = F.avg_pool2d(mask,mask.size()[2:])
        # mask = F.avg_pool2d(mask, (features.size(2)//2, features.size(3)//2))
        # mask = F.avg_pool2d(mask, 2)
        # features = F.avg_pool2d(features, kernel_size=5, stride=1, padding=2)
        # mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        features = F.avg_pool2d(features, kernel_size=5, stride=1, padding=2)
        mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        features = features * mask
        # features = F.interpolate(features, scale_factor=stride, mode='nearest')
        # features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        return features


    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def load_from_img_pretrained(self, state):
        self.load_state_dict(state, strict=False)
        for i in range(3, 9):
            getattr(self, f'layer{i}_1').load_state_dict({
                'tensor': torch.ones(getattr(self, f'layer{i}_1').tensor.shape),
            })
            for j in range(2, 4):
                getattr(self, f'layer{i}_{j}').load_state_dict({
                    'tensor': torch.zeros(getattr(self, f'layer{i}_1').tensor.shape),
                })
        print(self.__class__.__name__, 'model loaded')
        

def mbv2(num_classes=2, pretrained=False, **kwargs):
    """Constructs the network.

    Args:
        num_classes (int): the number of classes for the segmentation head to output.

    """
    model = MBv2(num_classes, **kwargs)
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
    net = mbv2(2, False).to(device)
    net.eval()
    x_last = torch.rand(1, 3, 512, 512, device=device)
    x = torch.rand(1, 3, 512, 512, device=device)
    mask = torch.rand(1, 1, 512, 512, device=device)
    feature = net.global_feature(x_last, mask)    # get last frame's features
    print([f.shape for f in feature])
    y = net(x,feature)# forward
    print(y.shape)
    # torch.onnx.export(net, x, "temp.onnx", verbose=False)
    exit(0)

    n_eval = 100
    tmp = 0
    for i in range(100):
        start_time = time.time()
        y = net(x,feature)
        end = time.time()
        tmp += (end - start_time)
    print(tmp / 100)
