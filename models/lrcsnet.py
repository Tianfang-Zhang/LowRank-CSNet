import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchsummary

from einops import rearrange, repeat

__all__ = ['LowRankCSNet']


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

#======================================================================================================================


class LowRankModule(nn.Module):
    def __init__(self, rank, channels=32, first_stage=False):
        super(LowRankModule, self).__init__()
        self.rank = rank
        self.channels = channels
        self.first_stage = first_stage
        self.lr_base = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        if not first_stage:
            self.lr_combine = nn.Sequential(
                nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
            )
        self.lr_pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((33, rank)),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )
        self.lr_pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((rank, 33)),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, Xk, Zk, feats):
        b, c, hei, wid = Xk.shape
        hc, wc = hei // 33, wid // 33

        # update L
        L = self.lr_base(torch.cat((Xk, Zk), dim=1))
        if not (self.first_stage and feats == None):
            L = self.lr_combine(torch.cat((L, feats), dim=1))

        tmp = rearrange(L, 'b c (h hc) (w wc) -> (b hc wc) c h w', h=33, w=33)
        L = torch.matmul(self.lr_pool1(tmp), self.lr_pool2(tmp))
        L = rearrange(L, '(b hc wc) c h w -> b c (h hc) (w wc)', h=33, w=33, hc=hc, wc=wc)
        return L


class DenseBlock(nn.Module):
    def __init__(self, g_channel=32, dense_layers=3):
        super(DenseBlock, self).__init__()
        self.g_channel = g_channel
        self.dense_layers = dense_layers

        self.convs = nn.ModuleList()
        for i in range(dense_layers):
            self.convs.append(nn.Sequential(
                nn.Conv2d(g_channel * (i + 1), g_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
            ))
        self.conv1 = nn.Conv2d(g_channel*self.dense_layers + g_channel, g_channel, 1, 1, 0)

    def forward(self, x):
        for i in range(self.dense_layers):
            rst = self.convs[i](x)
            x = torch.cat((rst, x), dim=1)
        x = self.conv1(x)

        return x


class DenseModule(nn.Module):
    def __init__(self, g_channel=32, dense_blocks=2, dense_layers=3):
        super(DenseModule, self).__init__()
        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.g_channel = g_channel

        self.blocks = nn.ModuleList()
        for i in range(dense_blocks):
            self.blocks.append(DenseBlock(g_channel=g_channel, dense_layers=dense_layers))

        self.conv = nn.Sequential(
            nn.Conv2d(g_channel * dense_blocks + g_channel, g_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, x):
        feats = [x]
        for i in range(self.dense_blocks):
            x = self.blocks[i](x)
            feats.append(x)
        out = self.conv(torch.cat(feats, dim=1))
        return out


class DeltaGModule(nn.Module):
    def __init__(self, g_channel=32, dense_blocks=2, dense_layers=3, first_stage=False):
        super(DeltaGModule, self).__init__()
        self.dense_blocks = dense_blocks
        self.dense_layers = dense_layers
        self.g_channel = g_channel
        self.first_stage = first_stage

        self.conv_up = nn.Sequential(
            nn.Conv2d(2, g_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(True)
        )

        if not first_stage:
            self.conv_integrate = nn.Sequential(
                nn.Conv2d(2 * g_channel, g_channel, kernel_size=1, padding=0, stride=1),
                nn.ReLU(True),
            )
            self.conv_transmission = nn.Sequential(
                nn.Conv2d(2 * g_channel, g_channel, kernel_size=1, padding=0, stride=1),
                nn.ReLU(True),
            )

        self.dense = DenseModule(g_channel=g_channel, dense_blocks=dense_blocks, dense_layers=dense_layers)
        self.conv_down = nn.Conv2d(g_channel, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, Zk, L, feats):
        feats_zl = self.conv_up(torch.cat((Zk, L), dim=1))
        if self.first_stage and feats == None:
            feats_new = self.dense(feats_zl)
            feats_tran = feats_new
        else:
            feats_new = self.conv_integrate(torch.cat((feats_zl, feats), dim=1))
            feats_new = self.dense(feats_new)
            feats_tran = self.conv_transmission(torch.cat((feats, feats_new), dim=1))
        rst_g = self.conv_down(feats_new)
        return rst_g, feats_tran


class ReconStage(nn.Module):
    def __init__(self, g_channel=32, rank=6, first_stage=False):
        super(ReconStage, self).__init__()
        self.g_channel = g_channel
        self.rank = rank
        self.first_stage = first_stage

        # parameter for L
        self.lrm = LowRankModule(rank=rank, channels=32, first_stage=first_stage)

        # parameter for Z
        self.rho1 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.rho2 = nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.eta = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

        # parameter for X
        self.alpha = nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.gamma = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.deltag = DeltaGModule(g_channel=g_channel, dense_blocks=2, dense_layers=3, first_stage=first_stage)

    def forward(self, Xk, Zk, feats, PhiWeight, PhiTWeight, PhiTb):
        # update L
        L = self.lrm(Xk, Zk, feats)

        # update Z
        Zk = self.rho1 * Xk + self.rho2 * Zk + (1 - self.rho1 - self.rho2) * L \
             - self.eta * PhiTPhi_fun(Xk, PhiWeight, PhiTWeight) + self.eta * PhiTb

        # update X
        rst_g, feats = self.deltag(Zk, L, feats)
        Xk = self.alpha * Xk + (1 - self.alpha) * Zk - self.gamma * rst_g

        return Xk, Zk, feats


class LowRankCSNet(nn.Module):
    '''
    Based on Lowrankcsnet41.

    Objective function:
    min_x ||y - phi x||^F_2 + lambda / 2 ||psi x||_1 + mu / 2 ||x - L||^F_2  s.t. z = x

    Update Z: Z = rho1 x^{k-1} + rho2 z^{k-1} + (1 - rho1 - rho2) L - eta phi^T phi x^{k-1} + eta phi^T y
    Update X: X = Z^k - gamma dG(Z^k, X^k, L)
    '''
    def __init__(self, stage_num, n_input, g_channel=32, rank=6, **kwargs):
        super(LowRankCSNet, self).__init__()

        self.n_input = n_input
        self.n_output = 1089
        self.stage_num = stage_num
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        self.reconstages = nn.ModuleList()
        for i in range(self.stage_num):
            first_stage = True if i == 0 else False
            self.reconstages.append(ReconStage(g_channel=g_channel, rank=rank, first_stage=first_stage))

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        MyBinarize = MySign.apply

        # Sampling-subnet
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        PhiWeight = Phi.contiguous().view(self.n_input, 1, 33, 33)
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=33, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)

        Xk = Zk = PhiTb
        feats = None

        # Recovery-subnet
        for i in range(self.stage_num):
            Xk, Zk, feats = self.reconstages[i](Xk, Zk, feats, PhiWeight, PhiTWeight, PhiTb)

        x_final = Xk

        eye = torch.eye(self.n_input).to(Phi.device)
        loss_phi = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1)) - eye, 2))
        return [x_final, loss_phi]
