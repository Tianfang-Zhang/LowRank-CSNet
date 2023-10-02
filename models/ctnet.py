import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torchsummary


__all__ = ['CTNet']


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


class Recovery(torch.nn.Module):
    def __init__(self, stage, hem_layers=6, hem_channel=32):  # stage = 1, 2, 3
        super(Recovery, self).__init__()

        self.rho_step = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.lambda_step = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)

        hem = [nn.Conv2d(stage+1, hem_channel, kernel_size=3, padding=1, stride=1),
                   nn.ReLU(True)]
        for i in range(hem_layers):
            hem.append(nn.Conv2d(hem_channel, hem_channel, kernel_size=3, padding=1, stride=1))
            hem.append(nn.ReLU(True))
        hem.append(nn.Conv2d(hem_channel, 1, kernel_size=3, padding=1, stride=1))
        self.cga = nn.Sequential(*hem)

    def forward(self, x, PhiWeight, PhiTWeight, PhiTb, prev_feats):
        out = x - self.rho_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        out = out + self.rho_step * PhiTb
        out = out - self.lambda_step * self.cga(torch.cat((out, prev_feats), dim=1))
        return out


class CTNet(torch.nn.Module):
    def __init__(self, stage_num, n_input, hem_layers=2, hem_channel=32, **kwargs):
        super(CTNet, self).__init__()

        self.n_input = n_input
        self.n_output = 1089
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        self.recon = nn.ModuleList()
        self.stage_num = stage_num

        for i in range(stage_num):
            self.recon.append(Recovery(stage=i+1, hem_layers=hem_layers, hem_channel=hem_channel))

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
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=33, bias=None)    # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb    # Conduct initialization
        feats = x

        # Recovery-subnet
        # layers_sym = []   # for computing symmetric loss
        for i in range(self.stage_num):
            x = self.recon[i](x, PhiWeight, PhiTWeight, PhiTb, feats)
            feats = torch.cat((feats, x), dim=1)

        x_final = x

        eye = torch.eye(self.n_input).to(Phi.device)
        loss_phi = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1)) - eye, 2))

        return [x_final, loss_phi]


if __name__ == '__main__':
    net = CTNet(stage_num=9, n_input=272, hem_layers=6, hem_channel=32)
    net.eval()
    print(net)

    img = torch.ones((3, 1, 99, 99))
    out = net(img)

    torchsummary.summary(net, (1, 99, 99))

    num_params = 0
    for para in net.parameters():
        num_params += para.numel()
    print("total para num: %d\n" % num_params)

    print(out[0].shape)
    print(out[1])