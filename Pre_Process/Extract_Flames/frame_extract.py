import cv2
import os

if __name__ == "__main__":
    vc = cv2.VideoCapture('D:/Sketch/Images/Videos/1.mp4')

    if vc.isOpened():
        print("open success")
    else:
        print("open failed")
        exit(-1)

    savedpath = 'D:/Sketch/Images/CO/'
    isExists = os.path.exists(savedpath)
    if not isExists:
        os.makedirs(savedpath)
        print("create dir")
    else:
        print("dir exists")

    c = 1
    while True:
        ok, frame = vc.read()
        if ok:
            cv2.imwrite(savedpath + str(c) + '.jpg', frame)
            c = c + 1
        else:
            break

    print("end")
    vc.release()
