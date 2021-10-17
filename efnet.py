
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet

from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain




class Self_attention(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet50', pretrained=True, freeze_bn=False,
                 freeze_backbone=False,use_aux=True):
        super(Self_attention, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.use_aux = use_aux
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer, dilated=True, multi_grid=True,
                                                 deep_base=True)
        m_out_sz = model.fc.in_features  # 2048



        self.initial = nn.Sequential(*list(model.children())[:4])  #

        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # 3x3卷积降维
        self.after_bacbone = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU())

        self.head1 = Head3(512, 256, reduction=4, h=64)  # 这个数等于输入图片数/8
        self.head2 = Head3(256, 128, reduction=4, h=64,softmax=True )

        self.value = nn.Sequential(nn.Conv2d(512, 128, 3,padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),)
        self.reduce = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.last = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(128, num_classes, 1))

        initialize_weights(self.head1, self.head2,self.last,self.after_bacbone)

        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        # print('inputsize', x.size())
        h, w = x.size()[2], x.size()[3]
        input_size = x.size()
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)  #  128*28*28
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        x1 = self.after_bacbone(x)

        x2 = self.head1(x1)

        x2 = self.head2(x2)


        x_value = self.value(x1)
        x3 = torch.mul(x2,x_value)  # 计算attention

        x = self.reduce(x1)  # x
        x =  self.gamma * x3 + x

        x = self.last(x)
        output =F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

        return output

    def get_backbone_params(self):
        return chain(self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters(),
                     self.layer4.parameters(), self.initial.parameters())

    def get_decoder_params(self):
        return chain(self.after_bacbone.parameters(), self.head1.parameters(), self.head2.parameters(),
                     self.last.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




class Head2(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, h=64,softmax=False):
        super(Head2, self).__init__()
        self.h = h
        self.sof = softmax
        self.query =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False))
        self.key =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False))
        #self.value = nn.Conv2d(in_channels, out_channels, 1)
        #self.reduce = nn.Conv2d(in_channels, out_channels, 1)

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.linear_h = nn.Sequential(
            nn.Linear(self.h, self.h * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.h * reduction, self.h, bias=False),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Dropout(0.1),

            #
        )
        self.linear_w = nn.Sequential(
            nn.Linear(self.h, self.h * reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.h * reduction, self.h, bias=True),
            nn.ReLU(inplace=True),
            #nn.Sigmoid(),
            nn.Dropout(0.1),
            #
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
       # value = self.value(x)
        # reduce = self.reduce(x)

        b, c, h, w = query.size()
        # print(b, c, h, w)
        q_h = self.pool_h(query)
        k_w = self.pool_w(key)

        q_h = q_h.view(b, c, -1)
        # print('linearqh',q_h.size())
        q_h = self.linear_h(q_h)  # b c h
        k_w = k_w.view(b, c, -1)
        k_w = self.linear_w(k_w)  # b c w

        q_h = q_h[:, :, :, None]  # 增加维度 调整形状做矩阵乘法
        q_h = q_h.view(b * c, h, -1)
        k_w = k_w[:, :, None, :]
        k_w = k_w.view(b * c, -1, w)

        energy = torch.bmm(q_h, k_w).view(b, c, h, w)  # b c h w
        if self.sof:
            attention = self.softmax(energy)
            out = attention
        else:
        #out = torch.mul(attention, value)                   .view(b, c, h, w)
        #out = self.gamma * out + x
            out =energy
        return out


class Head3(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, h=64,softmax=False):
        super(Head3, self).__init__()
        self.h = h
        self.sof = softmax
        self.query =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        self.key =  nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))


        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))

        self.linear_h = nn.Sequential(
            nn.Conv2d(out_channels,out_channels//reduction,1,bias=False),
            nn.BatchNorm2d(out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels //reduction,out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),


            #
        )
        self.linear_w = nn.Sequential(
            nn.Conv2d(out_channels,out_channels//reduction,1,bias=False),
            nn.BatchNorm2d(out_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//reduction,out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)

        b, c, h, w = query.size()
        #print(b, c, h, w)
        q_h = self.pool_h(query)
        k_w = self.pool_w(key)

        q_h = self.linear_h(q_h)  # b c h 1
        k_w = self.linear_w(k_w)  # b c 1 w

        q_h = q_h.view(b * c, w, -1)
        k_w = k_w.view(b * c, -1, h)

        energy = torch.bmm(q_h, k_w).view(b, c, h, w)  # b c h w
        if self.sof:
            attention = self.softmax(energy)
            out = attention
        else:
        #out = torch.mul(attention, value)                   .view(b, c, h, w)
        #out = self.gamma * out + x
            out =energy
        return out
# """
if __name__ == '__main__':
    from thop import profile

    r = Self_attention(pretrained=True, num_classes=17)
    inputs = torch.randn((1, 3, 248, 248))
    print(r)
    o = r(inputs)
    flops, params = profile(r, (inputs,))
    print('flops: ', flops, 'params: ', params)