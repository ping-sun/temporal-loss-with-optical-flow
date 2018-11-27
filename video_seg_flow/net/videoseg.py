import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import time
from .lw_mobilenet import mbv2 as network

class videobranch_diff(nn.Module):
    def __init__(self,  pretrained_model, background = False):
        super(videobranch_diff, self).__init__()
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.layer5 = pretrained_model.layer5
        self.layer6 = pretrained_model.layer6
        self.layer7 = pretrained_model.layer7
        self.layer8 = pretrained_model.layer8
        self.conv3 = pretrained_model.conv3
        self.conv4 = pretrained_model.conv4
        self.conv5 = pretrained_model.conv5
        self.conv6 = pretrained_model.conv6
        self.conv7 = pretrained_model.conv7
        self.conv8 = pretrained_model.conv8
        self.crp1 = pretrained_model.crp1
        self.crp2 = pretrained_model.crp2
        self.crp3 = pretrained_model.crp3
        self.crp4 = pretrained_model.crp4
        self.conv_adapt2 = pretrained_model.conv_adapt2
        self.conv_adapt3 = pretrained_model.conv_adapt3
        self.conv_adapt4 = pretrained_model.conv_adapt4
        self.relu = nn.ReLU6(inplace=True)
        self.bg = background
        if not background:
            self.layer1[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.layer1[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) if not background else nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    def forward(self, x):
        H, W = x.shape[2:]

        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=True)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=True)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=True)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)
        return l3

class videobranch_same(nn.Module):
    def __init__(self,  pretrained_model, background = False):
        super(videobranch_same, self).__init__()
        self.layer1 = pretrained_model.layer1[1:]
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.layer5 = pretrained_model.layer5
        self.layer6 = pretrained_model.layer6
        self.layer7 = pretrained_model.layer7
        self.layer8 = pretrained_model.layer8
        self.conv3 = pretrained_model.conv3
        self.conv4 = pretrained_model.conv4
        self.conv5 = pretrained_model.conv5
        self.conv6 = pretrained_model.conv6
        self.conv7 = pretrained_model.conv7
        self.conv8 = pretrained_model.conv8
        self.crp1 = pretrained_model.crp1
        self.crp2 = pretrained_model.crp2
        self.crp3 = pretrained_model.crp3
        self.crp4 = pretrained_model.crp4
        self.conv_adapt2 = pretrained_model.conv_adapt2
        self.conv_adapt3 = pretrained_model.conv_adapt3
        self.conv_adapt4 = pretrained_model.conv_adapt4
        self.relu = nn.ReLU6(inplace=True)
        self.bg = background
        #if not background:
        #    self.layer1[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.layer1[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) if not background else nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    def forward(self, x):
        H, W = x.shape[2:]

        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=True)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=True)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=True)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)
        return l3
        #out1 = F.avg_pool2d(l3, kernel_size=l3.size()[2:])
        #_,_,H,W = l3.size()
        #out1 = F.upsample(out1, size=(H,W), mode='bilinear')
        #if not self.bg:
        #    return l3, out1
        #else:
        #    return out1
        #out_segm = self.segm(l3)
        #return F.interpolate(out_segm, size=(H, W), mode='bilinear', align_corners=True)

class videobranch_resnet18(nn.Module):
    def __init__(self, background = False, pretrained_model = torchvision.models.resnet18(pretrained=True)):
        super(videobranch, self).__init__()
        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False) if not background else nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = pretrained_model.bn1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.bg = background
    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out1 = F.avg_pool2d(out, kernel_size=out.size()[2:])
        _,_,H,W = out.size()
        out1 = F.upsample(out1, size=(H,W), mode='bilinear')
        if not self.bg:
            return out, out1
        else:
            return out1


class hourglass(nn.Module):
    def __init__(self):
        super(hourglass, self).__init__()
        self.layer1 = nn.Conv2d(11, 16, kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2 * 2, stride=2, padding=2 // 2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2 * 3, stride=2, padding=4 // 2)
        self.deconv3 = nn.ConvTranspose2d(16, 11, kernel_size=2 * 5, stride=2, padding=8 // 2)
        self.changechan = nn.Conv2d(11, 2, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        y1 = x2 + self.deconv1(x3)
        y2 = x1 + self.deconv2(y1)
        y3 = x + self.deconv3(y2)
        y3 = self.changechan(y3)
        return y3

class Videoseg_same(nn.Module):
    def __init__(self,pretrained_model = network(2), background = False):
        super(Videoseg_same, self).__init__()
        #self.l1 = videobranch(pretrained_model = network(2))
        self.layer1 = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer2 = pretrained_model.layer1[0]
        self.bg = background

        #self.layer2[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.l2 = videobranch_same(pretrained_model = network(2))
        self.convch = nn.Conv2d(256*3, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.glass = hourglass()
    def forward(self, x1, x2):
        y1 = self.layer1(x1)
        x2 = self.layer2(x2)
        y1 = self.l2(y1)
        y2 = self.l2(x2)

        out1 = F.avg_pool2d(y1, kernel_size=y1.size()[2:])
        _,_,H,W = y1.size()
        out1 = F.upsample(out1, size=(H,W), mode='bilinear')

        y = torch.cat([y1,y2,out1],1)
        _,_,H,W = y.size()
        yo = F.upsample(y, size=(4*H,4*W), mode='bilinear')
        yo = self.convch(yo)

        #yo = F.softmax(y)

        yout = self.glass(torch.cat([yo,x1],1))
        #_,_,H,W = yout.size()
        #yout = F.upsample(yout, size=(32*H,32*W), mode='bilinear')
        return yo, yout

class Videoseg_diff(nn.Module):
    def __init__(self,pretrained_model = network(2), background = False):
        super(Videoseg_diff, self).__init__()
        self.l1 = videobranch_diff(pretrained_model = network(2),background = False)
        #self.layer1 = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.layer2 = pretrained_model.layer1[0]
        self.bg = background

        #self.layer2[0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.l2 = videobranch_diff(pretrained_model = network(2),background = True)
        self.convch = nn.Conv2d(256*3, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.glass = hourglass()
    def forward(self, x1, x2):
        #y1 = self.layer1(x1)
        #x2 = self.layer2(x2)
        y1 = self.l1(x1)
        y2 = self.l2(x2)

        out1 = F.avg_pool2d(y1, kernel_size=y1.size()[2:])
        _,_,H,W = y1.size()
        out1 = F.upsample(out1, size=(H,W), mode='bilinear')

        y = torch.cat([y1,y2,out1],1)
        _,_,H,W = y.size()
        yo = F.upsample(y, size=(4*H,4*W), mode='bilinear')
        yo = self.convch(yo)

        #yo = F.softmax(y)

        yout = self.glass(torch.cat([yo,x1],1))
        #_,_,H,W = yout.size()
        #yout = F.upsample(yout, size=(32*H,32*W), mode='bilinear')
        return yo, yout


class imagebranch(nn.Module):
    def __init__(self, pretrained_model = torchvision.models.resnet18(pretrained=True)):
        super(imagebranch, self).__init__()
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.finallayer = nn.Conv2d(512, 2, 1)
    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.finallayer(self.layer4(self.layer3(self.layer2(self.layer1(out)))))
        _,_,H,W = out.size()
        out = F.upsample(out, size=(32*H,32*W), mode='bilinear')
        return out




if __name__ == '__main__':
    from torch.autograd import Variable
    cudnn.benchmark = True

    #net = videobranch()
    #fms1 = net(Variable(torch.randn(1, 9, 512, 512)))

    #print (fms1[0].shape,fms1[1].shape)
    #net = videobranch(background = True)
    #fms2 = net(Variable(torch.randn(1, 6, 512, 512)))
    #print (fms2.shape)


    net = Videoseg().cuda()

    fms = net(Variable(torch.randn(2, 9, 480, 640)).cuda(async=True),Variable(torch.randn(2, 3, 480, 640)).cuda(async=True))
    net.eval()
    start_time = time.time()
    fms = net(Variable(torch.randn(2, 9, 480, 640)).cuda(async=True),Variable(torch.randn(2, 3, 480, 640)).cuda(async=True))
    end = time.time()

    print (fms[0].shape, fms[1].shape, end-start_time)

    net = hourglass().cuda()

    fms = net(Variable(torch.randn(2, 11, 480, 640)).cuda(async=True))
    net.eval()
    start_time = time.time()
    fms = net(Variable(torch.randn(2, 11, 480, 640)).cuda(async=True))
    end = time.time()

    print (fms.shape, end-start_time)
