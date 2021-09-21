import os
from torch.utils.data import DataLoader
import torch.utils.data as data
from data.data_utils import is_image_file, get_img, get_keras, get_distance_map


class My2dDataset(data.Dataset):
    def __init__(self, keras_path, gt_path, img_size, ref_num=2):
        super(My2dDataset, self).__init__()
        self.keras_path = keras_path
        self.gt_path = gt_path
        self.img_size = img_size
        self.ref_num = ref_num

        self.image_filenames = [x for x in os.listdir(self.gt_path) if is_image_file(x)]
        self.image_filenames.sort()

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        first_name = img_name.split('_')[0] + '_' + img_name.split('_')[1] + '_0001.' + img_name.split('.')[-1]
        last_name = None
        i = self.image_filenames.index(img_name)
        while True:
            temp_name = self.image_filenames[i + 1]
            if temp_name.split('_')[0] + '_' + temp_name.split('_')[1] != \
                    img_name.split('_')[0] + '_' + img_name.split('_')[1]:
                last_name = self.image_filenames[i]
                break
            i += 1
        img_gt = get_img(self.gt_path, img_name, self.img_size)
        first_gt = get_img(self.gt_path, first_name, self.img_size)
        last_gt = get_img(self.gt_path, last_name, self.img_size)
        img_keras = get_keras(self.keras_path, img_name, self.img_size)
        first_keras = get_keras(self.keras_path, first_name, self.img_size)
        last_keras = get_keras(self.keras_path, last_name, self.img_size)
        img_map = get_distance_map(self.keras_path, img_name, self.img_size)
        first_map = get_distance_map(self.keras_path, first_name, self.img_size)
        last_map = get_distance_map(self.keras_path, last_name, self.img_size)

        return img_keras, img_map, img_gt, [first_keras, last_keras], \
               [first_map, last_map], [first_gt, last_gt]

    def __len__(self):
        return len(self.image_filenames)


class My3dDataset(data.Dataset):
    def __init__(self, keras_path, fake_path, gt_path, img_size, ref_num=2):
        super(My3dDataset, self).__init__()
        self.keras_path = keras_path
        self.fake_path = fake_path
        self.gt_path = gt_path
        self.img_size = img_size
        self.ref_num = ref_num

        self.image_filenames = [x for x in os.listdir(self.fake_path) if is_image_file(x)]
        self.image_filenames.sort()

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        first_name = img_name.split('_')[0] + '_' + img_name.split('_')[1] + '_0001.' + img_name.split('.')[-1]
        last_name = None
        i = self.image_filenames.index(img_name)
        while True:
            temp_name = self.image_filenames[i + 1]
            if temp_name.split('_')[0] + '_' + temp_name.split('_')[1] != \
                    img_name.split('_')[0] + '_' + img_name.split('_')[1]:
                last_name = self.image_filenames[i]
                break
            i += 1
        img_gt = get_img(self.gt_path, img_name, self.img_size)
        first_gt = get_img(self.gt_path, first_name, self.img_size)
        last_gt = get_img(self.gt_path, last_name, self.img_size)
        img_keras = get_keras(self.keras_path, img_name, self.img_size)
        first_keras = get_keras(self.keras_path, first_name, self.img_size)
        last_keras = get_keras(self.keras_path, last_name, self.img_size)
        img_fake = get_img(self.fake_path, img_name, self.img_size)

        return img_keras, img_fake, img_gt, [first_keras, last_keras], [first_gt, last_gt]

    def __len__(self):
        return len(self.image_filenames)


def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = DataLoader(
            dataset=sample_dataset,
            batch_size=sample_size,
            drop_last=True,
            shuffle=True
        )

        for item in sample_loader:
            yield item
