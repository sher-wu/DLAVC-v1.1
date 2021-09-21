from PIL import Image
from scipy import ndimage
import os
import glob
import cv2


if __name__ == "__main__":
    img2path = 'D:/Sketch/Images/DF/'
    isExists = os.path.exists(img2path)
    if not isExists:
        os.makedirs(img2path)
        print("create dir")
    else:
        print("dir exists")

    threshold = 1

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    for files in glob.glob('D:/Sketch/Images/SK/*.jpg'):
        img = Image.open(files)

        _, imgName = os.path.split(files)

        output = ndimage.distance_transform_edt(img)
        output = cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(img2path + imgName, output)
        print(img2path + imgName)
        print(imgName + ' finished')
