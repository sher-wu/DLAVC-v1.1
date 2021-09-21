import os
import torch


def save_state(state, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path)
    print("=> saving checkpoint of epoch " + str(epoch))
    torch.save(state, os.path.join(path, 'params_' + str(epoch) + '.pth'))
    print("saving completed.")


def load_state(path, netG, netD, optimizerG, optimizerD):
    assert os.path.isfile(path)
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['state_dictG'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    netD.load_state_dict(checkpoint['state_dictD'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    epoch = checkpoint['epoch']
    count = checkpoint['count']
    print("loading completed.")
    return epoch, count


def log_train_data(log_info, config):
    log_dir = config.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "trainlogs.dat")
    with open(log_file, 'a') as f:
        f.write(' '.join([str(item[1]) for item in log_info]) + '\n')


# To create sample to show up. can be deleted.
def sample_2d(count, sample_iterator, netG, device, writer, data_type):
    kera, first_kera, last_kera, first_gt, last_gt, gt = next(sample_iterator)
    kera, first_kera, last_kera, first_gt, last_gt, gt = kera.to(device), first_kera.to(device), last_kera.to(device), \
                                                         first_gt.to(device), last_gt.to(device), gt.to(device)

    seq_len, img_size = kera.shape[1], kera.shape[-1]
    with torch.no_grad():
        fake = netG(kera, [first_kera, last_kera], [first_gt, last_gt])

    for i in range(fake.shape[0]):
        print('saving ' + str(data_type) + ' sample...')
        add_image_2d(first_kera[i].mul(0.5).add(0.5),
                  first_gt[i].mul(0.5).add(0.5),
                  last_kera[i].mul(0.5).add(0.5),
                  last_gt[i].mul(0.5).add(0.5),
                  kera[i].mul(0.5).add(0.5),
                  fake[i].mul(0.5).add(0.5),
                  gt[i].mul(0.5).add(0.5),
                  i, count, writer, data_type)


def sample_3d(count, sample_iterator, netG, writer, data_type):
    kera, fake, gt, ref_keras, ref_gts = next(sample_iterator)
    kera, fake, gt = kera.cuda(), fake.cuda(), gt.cuda()
    for i in range(len(ref_keras)):
        ref_keras[i] = ref_keras[i].cuda()
    for i in range(len(ref_gts)):
        ref_gts[i] = ref_gts[i].cuda()
    gts = gt.unsqueeze(1)
    for ref_gt in ref_gts:
        gts = torch.cat((gts, ref_gt.unsqueeze(1)), dim=1)
    with torch.no_grad():
        refine = netG(kera, fake, ref_keras, ref_gts)
    for i in range(fake.shape[0]):
        print('saving ' + str(data_type) + ' sample...')
        add_image_3d(refine[i].mul(0.5).add(0.5),
                     gts[i].mul(0.5).add(0.5),
                     i, count, writer, data_type)


def add_image_2d(first_kera, first_gt, last_kera, last_gt, kera, fake, gt, i, count, writer, data_type):
    cluster = torch.zeros((3, gt.shape[1], gt.shape[2] * 7))
    cluster[:, :, 0:gt.shape[2]] = first_kera
    cluster[:, :, gt.shape[2]:(gt.shape[1] * 2)] = first_gt
    cluster[:, :, (gt.shape[2] * 2):(gt.shape[1] * 3)] = last_kera
    cluster[:, :, (gt.shape[2] * 3):(gt.shape[1] * 4)] = last_gt
    cluster[:, :, (gt.shape[2] * 4):(gt.shape[1] * 5)] = kera
    cluster[:, :, (gt.shape[2] * 5):(gt.shape[1] * 6)] = fake
    cluster[:, :, (gt.shape[2] * 6):(gt.shape[1] * 7)] = gt

    writer.add_image(str(data_type) + '_' + str(i + 1), torch.clamp(cluster, 0, 1), count)


def cat_image(opti, seq_len, img_size):
    cat = torch.zeros((3, img_size, img_size * seq_len))
    offset = 0
    for i in range(seq_len):
        # print(i)
        # print(cat[:, :, offset:(offset + img_size)].shape)
        # print(opti[i].shape)
        cat[:, :, offset:(offset + img_size)] = opti[i]
        offset = offset + img_size
    return cat


def add_image_3d(fake, gt, i, count, writer, data_type):
    fake = cat_image(fake, fake.shape[0], fake.shape[-1])
    gt = cat_image(gt, gt.shape[0], gt.shape[-1])
    cluster = torch.zeros((3, gt.shape[1], gt.shape[2] * 2))
    cluster[:, :, 0:gt.shape[2]] = fake
    cluster[:, :, gt.shape[2]:(gt.shape[2] * 2)] = gt

    writer.add_image(str(data_type) + '_' + str(i + 1), torch.clamp(cluster, 0, 1), count)
