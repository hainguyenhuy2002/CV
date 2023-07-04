from super_image.modeling_utils import (
    default_conv,
    MeanShift,
    Upsampler,PreTrainedModel
)
from torch import nn

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class edsr(PreTrainedModel):
    
#     config_class = EdsrConfig
    
    def __init__(self, args, conv=default_conv):
        super(edsr, self).__init__(args)

        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_colors = args.n_colors
        kernel_size = 3
        scale = args.scale
        rgb_range = args.rgb_range
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # standardize input
        self.add_mean = MeanShift(rgb_range, sign=1, rgb_mean=args.rgb_mean, rgb_std=args.rgb_std)  # restore output

        # define head module, channels: 3->64
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module, channels: 64->64
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)

        return x