import os

import cv2
from PIL import Image
import torchvision.transforms as transforms
from scipy import ndimage


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_img(path, name, img_size):
    x = Image.open(os.path.join(path, name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x = transform(x)
    return x


def get_keras(path, name, img_size):
    x = Image.open(os.path.join(path, name)).convert('L')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    x = transform(x)
    return x


def get_distance_map(path, name, img_size):
    x = Image.open(os.path.join(path, name)).convert('L')
    x = ndimage.distance_transform_edt(x)
    x = cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
    x = Image.fromarray(x.astype('uint8')).convert("L")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    x = transform(x)
    return x