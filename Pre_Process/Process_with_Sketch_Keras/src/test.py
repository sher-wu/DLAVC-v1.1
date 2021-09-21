import numpy as np
import torch
import cv2
import glob
import os
from model import SketchKeras

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float64) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 3), dtype=np.float64)
    ret[0:h, 0:w, 0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert 1.0 >= thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":

    model = SketchKeras().to(device)
    model.load_state_dict(torch.load('D:/Sketch/Sketch_Keras/weights/model.pth'))
    print("model loaded")

    img2path = 'D:/Sketch/Images/SK/'
    isExists = os.path.exists(img2path)
    if not isExists:
        os.makedirs(img2path)
        print("create dir")
    else:
        print("dir exists")

    for files in glob.glob('D:/Sketch/Images/CO/*.jpg'):
        img = cv2.imread(files)
        _, imgname = os.path.split(files)

        # args = parse_args()

        # img = cv2.imread(args.input)

        # resize
        height, width = float(img.shape[0]), float(img.shape[1])
        if width > height:
            new_width, new_height = (512, int(512 / width * height))
        else:
            new_width, new_height = (int(512 / height * width), 512)
        img = cv2.resize(img, (new_width, new_height))

        # preprocess
        img = preprocess(img)
        x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
        x = torch.tensor(x).float()

        # feed into the network
        with torch.no_grad():
            pred = model(x.to(device))
        pred = pred.squeeze()

        # postprocess
        output = pred.cpu().detach().numpy()
        output = postprocess(output, thresh=0.1, smooth=False)
        output = output[:new_height, :new_width]

        cv2.imwrite(img2path + imgname, output)
        print(img2path + imgname)
        print(imgname + ' finished')
