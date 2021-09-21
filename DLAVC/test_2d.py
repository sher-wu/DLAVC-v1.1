import torch

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.networks import ColorTransformG, ColorTransformD
from util.utils import load_state
from data.data_loader import My2dDataset


if __name__ == '__main__':

    assert torch.cuda.is_available()
    device = torch.device('cuda')

    torch.manual_seed(1234)
    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    netG = torch.nn.DataParallel(ColorTransformG(2))
    netD = torch.nn.DataParallel(ColorTransformD())

    optimizerG = optim.Adam(netG.parameters(), lr=0, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0, betas=(0.5, 0.999))

    epoch = 1
    count = 1

    netD = netD.to(device)
    netG = netG.to(device)

    epoch, count = load_state('./checkpoint_2d/params_100.pth', netG, netD, optimizerG, optimizerD)

    test_set = My2dDataset('test/keras', 'test/frame', 256)

    test_data_loader = DataLoader(test_set, batch_size=2, shuffle=False, pin_memory=True, num_workers=4)

    for iteration, [kera, mp, gt, ref_keras, ref_mps, ref_gts] in enumerate(test_data_loader):
        path = "./outputs_2d/%d.jpg" % iteration
        kera, mp, gt = kera.to(device), mp.to(device), gt.to(device)
        for ref_kera in ref_keras:
            ref_kera = ref_kera.to(device)
        for ref_mp in ref_mps:
            ref_mp = ref_mp.to(device)
        for ref_gt in ref_gts:
            ref_gt = ref_gt.to(device)

        with torch.no_grad():
            fake, y_sim, y_mid = netG(kera, mp, ref_keras, ref_mps, ref_gts)

        fake = fake[0]
        save_image(fake, path)


