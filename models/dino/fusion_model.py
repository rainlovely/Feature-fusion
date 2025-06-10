import torch
import torch.nn as nn
import torch.nn.functional as F


class convNormlayer(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size,stride,padding=None,bias=False,act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self,x):
        return self.act(self.norm(self.conv(x)))


class Repblock(nn.Module):
    def __init__(self,ch_in,ch_out,act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.act = nn.Identity() if act is None else get_activation(act)
        self.conv1 = convNormlayer(ch_in,ch_out,1,1,act=None)
        self.conv2 = convNormlayer(ch_in,ch_out,3,1,act=None)
        self.conv3 = convNormlayer(ch_out,ch_in,3,1,act=None)
        self.conv4 = nn.Identity()

    def forward(self,x):
        x_1 = x
        x_2 = self.act(self.conv2(x) + self.conv1(x))
        x_2 = self.act(self.conv3(x_2))
        return self.conv4(x_1+x_2)

class CSPblock(nn.Module):
    def __init__(self,ch_in,ch_out,act='relu',num_blocks=3,expansion=0.5,bias=None):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        hidden_ch = int(ch_out * expansion)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.conv0 = convNormlayer(ch_in, ch_in, 1, 1, act=act)
        self.conv1 = convNormlayer(ch_in,ch_out,1,1,act=act)
        self.conv2 = convNormlayer(ch_in,hidden_ch,1,1,act=act)
        self.conv3 = convNormlayer(hidden_ch,ch_out,1,1,act=act)
        self.bottlenecks = nn.Sequential(*[Repblock(ch_in,hidden_ch,act=act) for _ in range(num_blocks)])
        if hidden_ch!=ch_out:
            self.conv4 = convNormlayer(ch_in*3+ch_out,ch_out,1,1,act=act)
        else:
            self.conv4 = nn.Identity()

    def forward(self,x):
        x_1 = self.conv1(x)
        x_2_inner = self.conv0(x)
        x_21 = self.bottlenecks[0](x_2_inner)
        x_22 = self.bottlenecks[1](x_2_inner)
        x_23 = self.bottlenecks[2](x_2_inner)
        x_3 = torch.cat([x_1,x_21,x_22,x_23],dim=1)
        return self.conv4(x_3)






def get_activation(act: str, inpace: bool = True):
    '''get activation
    '''
    ##act = act.lower()

    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inpace

    return m