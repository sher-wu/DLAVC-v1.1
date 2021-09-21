import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.encoder1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, 64, 3, 1, 1))
        self.encoder2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 2, 2))
        self.encoder3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 2, 2))
        self.ReLU = nn.ReLU()

    def forward(self, image):
        x = self.encoder1(image)  # [batch_size, in_channel, 256, 256]
        x = self.ReLU(x)
        x = self.encoder2(x)
        x = self.ReLU(x)
        x = self.encoder3(x)
        x = self.ReLU(x)
        return x  # [batch_size, 256, 64, 64]


class Decoder(nn.Module):
    def __init__(self, in_channel):
        super(Decoder, self).__init__()
        self.decoder1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, 256, 3, 1, 1))
        self.decoder2 = nn.utils.spectral_norm(nn.Conv2d(256, 128, 3, 1, 1))
        self.decoder3 = nn.utils.spectral_norm(nn.Conv2d(128, 64, 3, 1, 1))
        self.decoder4 = nn.utils.spectral_norm(nn.Conv2d(64, 3, 3, 1, 1))
        self.ReLU = nn.ReLU()
        self.Upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.decoder1(x)  # [batch_size, in_channel, 64, 64]
        x = self.ReLU(x)
        x = self.decoder2(x)
        x = self.Upsample(x)
        x = self.ReLU(x)
        x = self.decoder3(x)
        x = self.Upsample(x)
        x = self.ReLU(x)
        x = self.decoder4(x)
        image = self.ReLU(x)
        return image  # [batch_size, 3, 256, 256]


class SBCTL(nn.Module):
    def __init__(self, ref_num=3):
        super().__init__()
        self.ref_num = ref_num
        self.sigmoid = nn.Sigmoid()
        self.conv_dec_1 = nn.Conv2d(256, 32, 1)
        self.conv_dec_3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv_inc = nn.Conv2d(32, 256, 1)

    def forward(self, fd, fds, fys):
        M_i = []
        batch_size = fd.shape[0]
        for i in range(self.ref_num):
            fdi_temp = fds[i]  # (batch_size, 256, 64, 64)
            fdi_temp = self.conv_dec_1(fdi_temp)  # (batch_size, 32, 64, 64)
            fdi_temp = torch.reshape(fdi_temp, (fdi_temp.shape[0], fdi_temp.shape[1], -1))  # (batch_size, 32, 4096)

            fd_temp = self.conv_dec_1(fd)  # (batch_size, 32, 64, 64)
            fd_temp = torch.reshape(fd_temp, (fd_temp.shape[0], fd_temp.shape[1], -1))  # (batch_size, 32, 4096)
            fd_temp = fd_temp.permute(0, 2, 1)  # (batch_size, 4096, 32)

            M_i.append(torch.matmul(fd_temp, fdi_temp))

        M = torch.cat(M_i, dim=1)  # (batch_size, 4096*ref_num, 4096)

        m_i = []
        for i in range(self.ref_num):
            m_i.append(self.sigmoid(self.conv_dec_3(torch.cat((fd, fds[i]), dim=1))))  # (batch_size, 256, 64, 64)

        n_i = m_i

        C_i = []
        for i in range(self.ref_num):
            fyi_temp = fys[i]
            fmyi_temp = torch.mul(fyi_temp, m_i[i])
            fmyi_temp = self.conv_dec_1(fmyi_temp)  # (batch_size, 32, 64, 64)
            fmyi_temp = torch.reshape(fmyi_temp, (fmyi_temp.shape[0], fmyi_temp.shape[1], -1))  # (batch_size, 32, 4096)

            C_i.append(fmyi_temp)

        C = torch.cat(C_i, dim=2)  # (batch_size, 32, 4096*ref_num)

        f_mat = torch.matmul(C, M)  # (batch_size, 32, 4096)
        f_mat = torch.reshape(f_mat, (batch_size, 32, 64, 64))  # (batch_size, 32, 64, 64)
        f_mat = self.conv_inc(f_mat)

        f_sim_i = []
        for i in range(self.ref_num):
            f_sim_i.append(torch.mul(f_mat, 1 - n_i[i]) + torch.mul(n_i[i], fys[i]))

        f_sim = torch.mean(torch.stack(f_sim_i), 0)

        return f_sim  # [batch_size, 256, 64, 64]


class Embedder(nn.Module):
    def __init__(self, ref_num):
        super().__init__()
        self.ref_num = ref_num
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, 3, 1, 1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 2, 2)),
            nn.ReLU(),
            nn.AvgPool2d(16)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        )

    def forward(self, lines, refs):
        temp_i = []
        for i in range(self.ref_num):
            temp_i.append(self.conv(torch.cat((lines[i], refs[i]), dim=1)))
        temp = torch.mean(torch.stack(temp_i), 0).squeeze()
        p_em = self.fc(temp)
        return p_em


class LatentDecoder(nn.Module):
    def __init__(self, in_channel):
        super(LatentDecoder, self).__init__()
        self.conv = nn.Conv2d(in_channel, 3, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


class AdaINResBlocks(nn.Module):
    def __init__(self, dim, num_blocks=8):
        super(AdaINResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """

        :param x: [batch_size, dim, 256, 256]
        :return: [batch_size, dim, 256, 256]
        """

        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        model = [Conv2dBlock(dim, dim, 3, 1, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x
        out += self.model(x)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super(Conv2dBlock, self).__init__()
        # initialize padding
        self.pad = nn.ZeroPad2d(padding)
        # initialize normalization
        norm_dim = output_dim
        self.norm = AdaptiveInstanceNorm2d(norm_dim)
        # initialize activation
        self.activation = nn.ReLU()
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            # to check if padding is not a 0-d array, otherwise tuple(padding) will raise an exception
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class NN3Dby2D(object):
    '''
    Use these inner classes to mimic 3D operation by using 2D operation frame by frame.
    '''
    class Base(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, xs):
            dim_length = len(xs.shape)
            if dim_length == 5:  # [batch_size, channels, video_len, w, h]
                # Unbind the video data to a tuple of frames
                xs = torch.unbind(xs, dim=2)
                # Process them frame by frame using 2d layer
                xs = torch.stack([self.layer(x) for x in xs], dim=2)
            elif dim_length == 4:  # [batch_size, channels, w, h]
                # keep the 2D ability when the data is not batched videoes but batched frames
                xs = self.layer(xs)
            return xs

    class Conv3d(Base):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride, padding, dilation, groups, bias):
            super().__init__()
            # take off the kernel/stride/padding/dilation setting for the temporal axis
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[1:]
            if isinstance(stride, tuple):
                stride = stride[1:]
            if isinstance(padding, tuple):
                padding = padding[1:]
            if isinstance(dilation, tuple):
                dilation = dilation[1:]
            self.layer = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias
            )

            # let the spectral norm function get its conv weights
            self.weight = self.layer.weight
            # let partial convolution get its conv bias
            self.bias = self.layer.bias
            self.__class__.__name__ = "Conv3dBy2D"

    class BatchNorm3d(Base):
        def __init__(self, out_channels):
            super().__init__()
            self.layer = nn.BatchNorm2d(out_channels)

    class InstanceNorm3d(Base):
        def __init__(self, out_channels, track_running_stats=True):
            super().__init__()
            self.layer = nn.InstanceNorm2d(out_channels, track_running_stats=track_running_stats)


class NN3Dby2DTSM(NN3Dby2D):

    class Conv3d(NN3Dby2D.Conv3d):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride, padding, dilation, groups, bias):
            super().__init__(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias
            )
            self.__class__.__name__ = "Conv3dBy2DTSM"

        def forward(self, xs):
            # identity = xs
            B, C, L, H, W = xs.shape
            # Unbind the video data to a tuple of frames
            from models.tsm_utils import tsm
            xs_tsm = tsm(xs.transpose(1, 2), L, 'zero').contiguous()
            out = self.layer(xs_tsm.view(B * L, C, H, W))
            _, C_, H_, W_ = out.shape
            return out.view(B, L, C_, H_, W_).transpose(1, 2)

            '''
            # Process them frame by frame using 2d layer
            xs_tsm_unbind = torch.unbind(xs_tsm.transpose(1, 2), dim=2)
            xs_rebind = torch.stack([self.layer(x) for x in xs_tsm_unbind], dim=2)
            out = xs_rebind
            '''

            '''
            # This residual will cause errors due to the inconsistent channel numbers
            if xs_rebind.shape != identity.shape:
                scale_factor = [x / y for (x, y) in zip(xs_rebind.shape[2:5], identity.shape[2:5])]
                out = xs_rebind + F.interpolate(identity, scale_factor=scale_factor)
            else:
                out = xs_rebind + identity

            '''
            return out


class VanillaConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):

        super().__init__()
        if conv_by == '2d':
            self.module = NN3Dby2D
        elif conv_by == '2dtsm':
            self.module = NN3Dby2DTSM
        elif conv_by == '3d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')

        self.padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2) if padding == -1 else padding
        self.featureConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = self.module.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = self.module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_by='3d'
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class GatedConv(VanillaConv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.Tanh(), conv_by='2dtsm'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by
        )
        # print(in_channels)
        if conv_by == '2dtsm':
            self.module = NN3Dby2D
        self.gatingConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
        self.sigmoid = nn.Sigmoid()
        self.store_gated_values = False

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        out = self.sigmoid(mask)
        if self.store_gated_values:
            self.gated_values = out.detach().cpu()
        return out

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = (1 + self.gated(gating)) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(VanillaDeconv):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_by='3d'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, scale_factor, conv_by
        )
        self.conv = GatedConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
