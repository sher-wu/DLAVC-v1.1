import os

import torch.nn as nn
import torch
import argparse

import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.networks import ColorTransformG, ColorTransformD
from util.loss2d import StyleLoss2d, PerceptualLoss2d, AdversarialLoss2d, LatentconstraintLoss
from util.utils import load_state, log_train_data, sample_2d, save_state
from data.data_loader import My2dDataset, create_iterator


def set_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ref_num', type=int, default=2, help='')

    parser.add_argument('--train_gt_path', default='/data2/wn/Video_dataset/train/frame', help='')
    parser.add_argument('--train_kera_path', default='/data2/wn/Video_dataset/train/kera', help='')
    parser.add_argument('--test_gt_path', default='/data2/wn/Video_dataset/val/frame', help='')
    parser.add_argument('--test_kera_path', default='/data2/wn/Video_dataset/val/kera', help='')
    parser.add_argument('--board_path', default='./board_2d/', help='')
    parser.add_argument('--log_dir', default='./log_2d/', help='')
    parser.add_argument('--save_path', default='./checkpoint_2d/', help='')
    parser.add_argument('--checkpoint_path', required=False, help='')

    parser.add_argument('--epoch', type=int, default=500, help='')
    parser.add_argument('--log_freq', type=int, default=500, help='/iteration')
    parser.add_argument('--sample_freq', type=int, default=500, help='/iteration')
    parser.add_argument('--save_freq', type=int, default=1, help='/epoch')

    parser.add_argument('--seq_len', type=int, default=8, help='')
    parser.add_argument('--train_batch_size', type=int, default=4, help='')
    parser.add_argument('--sample_batch_size', type=int, default=2, help='')
    parser.add_argument('--img_size', type=int, default=256, help='')
    parser.add_argument('--workers', type=int, default=4, help='')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--base_G_lr', type=float, default=0.00001, help='')
    parser.add_argument('--base_D_lr', type=float, default=0.00001, help='')

    parser.add_argument('--L1lamb', type=float, default=10, help='')
    parser.add_argument('--Perceplamb', type=float, default=1, help='')
    parser.add_argument('--Stylelamb', type=float, default=1000, help='')
    parser.add_argument('--Ganlamb', type=float, default=1, help='')
    parser.add_argument('--Latentlamb', type=float, default=1, help='')
    return parser.parse_args()


if __name__ == '__main__':
    config = set_parser()

    assert torch.cuda.is_available()
    torch.manual_seed(config.seed)
    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    netG = torch.nn.DataParallel(ColorTransformG(config.ref_num))
    netD = torch.nn.DataParallel(ColorTransformD())

    optimizerG = optim.Adam(netG.parameters(), lr=config.base_G_lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=config.base_D_lr, betas=(0.5, 0.999))

    L1Loss = nn.L1Loss()
    PercepLoss = PerceptualLoss2d()
    StyleLoss = StyleLoss2d()
    GANLoss = AdversarialLoss2d()
    LatentLoss = LatentconstraintLoss()

    epoch = 1
    count = 1

    netD = netD.cuda()
    netG = netG.cuda()
    L1Loss = L1Loss.cuda()
    PercepLoss = PercepLoss.cuda()
    StyleLoss = StyleLoss.cuda()
    GANLoss = GANLoss.cuda()
    LatentLoss = LatentLoss.cuda()

    if config.checkpoint_path:
        epoch, count = load_state(config.checkpoint_path, netG, netD, optimizerG, optimizerD)

    train_set = My2dDataset(config.train_kera_path, config.train_gt_path, config.img_size)
    test_set = My2dDataset(config.test_kera_path, config.test_gt_path, config.img_size)

    training_data_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True,
                                      pin_memory=True, num_workers=config.workers)

    sample_train_iterator = create_iterator(config.sample_batch_size, train_set)
    sample_test_iterator = create_iterator(config.sample_batch_size, test_set)

    if not os.path.exists(config.board_path):
        os.makedirs(config.board_path)
    writer = SummaryWriter(config.board_path)

    for epoch in range(epoch, config.epoch + 1):
        for iteration, [kera, mp, gt, ref_keras, ref_mps, ref_gts] in enumerate(training_data_loader):

            netG.train()
            netD.train()

            kera, mp, gt = kera.cuda(), mp.cuda(), gt.cuda()
            for i in range(len(ref_keras)):
                ref_keras[i] = ref_keras[i].cuda()
            for i in range(len(ref_mps)):
                ref_mps[i] = ref_mps[i].cuda()
            for i in range(len(ref_gts)):
                ref_gts[i] = ref_gts[i].cuda()
            ############################
            # (1) train discriminator
            ############################

            for p in netD.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False

            netD.zero_grad()
            with torch.no_grad():
                fake, y_sim, y_mid = netG(kera, mp, ref_keras, ref_mps, ref_gts)

            pred_fake = netD(kera, fake)
            loss_D_fake = GANLoss(pred_fake, False, True)
            pred_gt = netD(kera, gt)
            loss_D_real = GANLoss(pred_gt, True, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            optimizerD.step()

            ############################
            # (2) train generator
            ############################

            for p in netD.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = True

            netG.zero_grad()

            fake, y_sim, y_mid = netG(kera, mp, ref_keras, ref_mps, ref_gts)
            with torch.no_grad():
                pred_fake = netD(kera, fake)

            loss_G_gan = GANLoss(pred_fake, True, False) * config.Ganlamb
            loss_G_l1 = L1Loss(fake, gt) * config.L1lamb
            loss_G_percep = PercepLoss(fake, gt) * config.Perceplamb
            loss_G_style = StyleLoss(fake, gt) * config.Stylelamb
            loss_G_latent = LatentLoss(y_sim, y_mid, gt) * config.Latentlamb
            loss_G = loss_G_gan + loss_G_l1 + loss_G_percep + loss_G_style + loss_G_latent

            loss_G.backward()
            optimizerG.step()
            count = count + 1

            ############################
            # (3) log & sample & save
            ############################

            if iteration % config.log_freq == 0:

                logs = [("epoc", epoch), ("iter", iteration),
                        ("Loss_D", loss_D.item()), ("Loss_D_Real", loss_D_real.item()),
                        ("Loss_D_Fake", loss_D_fake.item()), ("Loss_G", loss_G.item()),
                        ("Loss_G_adv", loss_G_gan.item()), ("Loss_G_L1", loss_G_l1.item()),
                        ("Loss_G_style", loss_G_style.item()), ("Loss_G_percep", loss_G_percep.item()),
                        ("loss_G_latent", loss_G_latent)]
                log_train_data(logs, config)
                writer.add_scalar("Loss_D", loss_D.item(), count)
                writer.add_scalar("Loss_D_Real", loss_D_real.item(), count)
                writer.add_scalar("Loss_D_Fake", loss_D_fake.item(), count)
                writer.add_scalar("Loss_G", loss_G.item(), count)
                writer.add_scalar("Loss_G_adv", loss_G_gan.item(), count)
                writer.add_scalar("Loss_G_L1", loss_G_l1.item(), count)
                writer.add_scalar("Loss_G_style", loss_G_style.item(), count)
                writer.add_scalar("Loss_G_percep", loss_G_percep.item(), count)
                writer.add_scalar("loss_G_latent", loss_G_latent, count)

            if iteration % config.sample_freq == 0:
                sample_2d(count, sample_train_iterator, netG.eval(), writer, "train")
                sample_2d(count, sample_test_iterator, netG.eval(), writer, "test")

            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} LossD_Fake: {:.4f} LossD_Real: {:.4f} "
                  "Loss_G: {:.4f} LossG_Adv: {:.4f} LossG_L1: {:.4f} "
                  "LossG_Style {:.4f} LossG_Percep {:.4f} loss_G_latent {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_D, loss_D_fake, loss_D_real, loss_G,
                loss_G_gan, loss_G_l1, loss_G_style, loss_G_percep, loss_G_latent))

        if epoch % config.save_freq == 0:

            save_state({'state_dictG': netG.state_dict(),
                        'optimizerG': optimizerG.state_dict(),
                        'state_dictD': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'epoch': epoch,
                        'count': count}, config.save_path, epoch)
