import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import Encoder, SBCTL, Embedder, AdaINResBlocks, Decoder, LatentDecoder, GatedConv, GatedDeconv


class ColorTransformG(nn.Module):
    def __init__(self, ref_num=2):
        super(ColorTransformG, self).__init__()
        self.ref_num = ref_num
        self.encoder_c = Encoder(3)
        self.encoder_l = Encoder(1)
        self.sbctl = SBCTL(self.ref_num)
        self.embedders = Embedder(self.ref_num)
        self.adain_res = AdaINResBlocks(dim=256)
        self.decoder = Decoder(256)
        self.sim_decoder = LatentDecoder(256)
        self.mid_decoder = LatentDecoder(256)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in models
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":

                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, x, d, ref_xs, drs, yrs):

        f_x_in = self.encoder_l(x)  # (batch_size, 256, 64, 64)
        f_d_in = self.encoder_l(d)  # (batch_size, 256, 64, 64)
        f_ds_in = [self.encoder_l(drs[i]) for i in range(self.ref_num)]  # (batch_size, 256, 64, 64)
        f_ys_in = [self.encoder_c(yrs[i]) for i in range(self.ref_num)]  # (batch_size, 256, 64, 64)

        # Similarity based Color Transform Layer
        f_sim = self.sbctl(f_d_in, f_ds_in, f_ys_in)  # (batch_size, 256, 64, 64)

        # Embedder
        p_em = self.embedders(ref_xs, yrs)  # (batch_size, 512)
        p_em = torch.reshape(p_em, (-1, 512))
        # AdaIN Resblocks
        self.assign_adain_params(p_em, self.adain_res)
        f_mid = self.adain_res(torch.add(f_x_in, f_sim))  # (batch_size, 256, 64, 64)

        y_sim = self.sim_decoder(f_sim)
        y_mid = self.mid_decoder(f_mid)
        pic = self.decoder(f_mid)

        return pic, y_sim, y_mid


# class ColorTransformG(nn.Module):
#     def __init__(self, ref_num):
#         super(ColorTransformG, self).__init__()
#         self.encoder_c = Encoder(3)
#         self.encoder_l = Encoder(1)
#         self.decoder = Decoder(256)
#
#     def forward(self, d, drs, yrs):
#         f_d_in = self.encoder_l(d)  # (batch_size, 256, 64, 64)
#         pic = self.decoder(f_d_in)
#
#         return pic, f_d_in, f_d_in


class ColorTransformD(nn.Module):
    def __init__(self, channel_in=4):
        super(ColorTransformD, self).__init__()
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channel_in, 64, 3, 1, 1)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 2, 2)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, 2, 2))
        )

    def forward(self, line, pic):
        """

        :param line: [batch_size, 1, 256, 256]
        :param pic: [batch_size, 3, 256, 256]
        :return:
        """

        x = torch.cat((line, pic), dim=1)
        return self.discriminator(x)


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvBlock = GatedConv
        self.DeconvBlock = GatedDeconv


class TemporalConstraintG(BaseModule):
    def __init__(self, nf=64, nc_in=4, nc_out=3, use_bias=True, norm='SN', conv_by='2dtsm'):
        super().__init__()
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 3, 3), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 2
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 4, 4))
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 8, 8))
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(1, 16, 16))

        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nf * 4 * 2, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        # Upsample 2
        self.deconv2 = self.DeconvBlock(
            nf * 2 * 2, nf, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv4 = self.ConvBlock(
            nf * 2, nc_out, kernel_size=(3, 3, 3), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

    def preprocess(self, img):
        return torch.transpose(img, 1, 2)

    def postprocess(self, img):
        return torch.transpose(img, 1, 2)

    def forward(self, kera, fake, ref_keras, ref_gts):
        """

        :param kera: [batch_size, channel(1), 256, 256]
        :param fake: [batch_size, channel(3), 256, 256]
        :param ref_keras: list -> ref_num * [batch_size, channel(1), 256, 256]
        :param ref_gts: list -> ref_num * [batch_size, channel(3), 256, 256]
        :return: [batch_size, ref_num + 1, channel(3), 256, 256]
        """
        inputs = [torch.cat([kera, fake], dim=1).unsqueeze(1)]
        for i in range(len(ref_keras)):
            inputs.append(torch.cat([ref_keras[i], ref_gts[i]], dim=1).unsqueeze(1))
        inputs = torch.cat(inputs, dim=1)
        inputs = self.preprocess(inputs)

        d1 = self.conv1(inputs)  # (batch_size, nf, seq_len, w, h)

        d2 = self.conv2(d1)  # (batch_size, nf * 2, seq_len, w / 2, h / 2)
        d3 = self.conv3(d2)  # (batch_size, nf * 4, seq_len, w / 4, h / 4)

        c1 = self.dilated_conv1(d3)  # (batch_size, nf * 4, seq_len, w / 4, h / 4)
        c2 = self.dilated_conv2(c1)  # (batch_size, nf * 4, seq_len, w / 4, h / 4)
        c3 = self.dilated_conv3(c2)  # (batch_size, nf * 4, seq_len, w / 4, h / 4)
        c4 = self.dilated_conv4(c3)  # (batch_size, nf * 4, seq_len, w / 4, h / 4)

        u1 = self.deconv1(torch.cat((c4, d3), 1))  # (batch_size, nf * 2, seq_len, w / 2, h / 2)
        u2 = self.deconv2(torch.cat((u1, d2), 1))  # (batch_size, nf, seq_len, w, h)
        u3 = self.conv4(torch.cat((u2, d1), 1))  # (batch_size, nc_out, seq_len, w, h)

        return self.postprocess(u3)


class TemporalConstraintD(BaseModule):
    def __init__(self, nf=64, nc_in=3, norm='SN', use_sigmoid=True, use_bias=True, conv_by='2dtsm'):
        super().__init__()
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
            padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.sigmoid = nn.Sigmoid()

    def preprocess(self, img):
        return torch.transpose(img, 1, 2)

    def postprocess(self, img):
        return torch.transpose(img, 1, 2)

    def forward(self, x):
        """

        :param x: [batch_size, ref_num + 1, channel(3), 256, 256]
        :return: [batch_size, ref_num + 1, 256, 8, 8]
        """

        x = self.preprocess(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        if self.use_sigmoid:
            c5 = torch.sigmoid(c5)

        return self.postprocess(c5)


if __name__ == '__main__':
    device = 'cpu'
    refnum = 2
    img_size = 256
    x = torch.randn(4, 1, img_size, img_size).to(device)
    d = torch.randn(4, 1, img_size, img_size).to(device)
    ref_xs = [torch.randn(4, 1, img_size, img_size).to(device) for i in range(refnum)]
    drs = [torch.randn(4, 1, img_size, img_size).to(device) for i in range(refnum)]
    yrs = [torch.randn(4, 3, img_size, img_size).to(device) for i in range(refnum)]

    net1 = nn.DataParallel(ColorTransformG(refnum)).to(device)
    net2 = nn.DataParallel(ColorTransformD()).to(device)
    netG = nn.DataParallel(TemporalConstraintG()).to(device)
    netD = nn.DataParallel(TemporalConstraintD()).to(device)
    print("Total number of paramerters in networks is {}  "
          .format(sum(x.numel() for x in net1.parameters())))
    # print(net1)
    # pic, y_sims, y_mids = net1(x, d, ref_xs, drs, yrs)
    # logit = net2(d, pic)
    fake = torch.randn(4, 3, img_size, img_size).to(device)
    ys = netG(x, fake, drs, yrs)
    # print(pic.shape)
    # print(logit.shape)
    # print(y_sims.shape)
    # print(y_mids.shape)


    # input = []
    # input.append(torch.cat((pic, d), dim=1).unsqueeze(1))
    # for i in range(refnum):
    #     input.append(torch.cat((drs[i], yrs[i]), dim=1).unsqueeze(1))
    # input = torch.cat(input, dim=1)
    #
    # imgs = netG(input)
    # logit = netD(input)
    #
    # print(imgs.shape)
    # print(logit.shape)
