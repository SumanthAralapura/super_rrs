import cv2
from numpy import asarray
from datetime import datetime

print('cv2 version', cv2.__version__)


def espcn(np_img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "ESPCN_x4.pb"
    sr.readModel(path)
    sr.setModel("espcn", 4)
    result = sr.upsample(np_img)  # upscale the input image
    cv2.imwrite('output/test2_ESPCN_x4' + datetime.now() + '.jpg', result)


def edsr(np_img):
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "EDSR_x4.pb"
        sr.readModel(path)
        sr.setModel("edsr", 4)
        result = sr.upsample(np_img)
        cv2.imwrite('output/test_EDSR_x4' + datetime.now() + '.jpg', result)
    except:
        print('error in edsr')


def fsrcnn(np_img):
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "FSRCNN_x4.pb"
        sr.readModel(path)
        sr.setModel("fsrcnn", 3)
        result = sr.upsample(np_img)
        cv2.imwrite('output/test_FSRCNN_x4' + datetime.now() + '.jpg', result)
    except:
        print('fsrcnn error')


def lapsrn(np_img):
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "LapSRN_x8.pb"
        sr.readModel(path)
        sr.setModel("lapsrn", 8)
        result = sr.upsample(np_img)
        cv2.imwrite('output/test_LapSRN_x8' + datetime.now() + '.jpg', result)
    except:
        print('error in lapSRN')


if __name__ == '__main__':
    img = cv2.imread("data/test2.png")
    espcn(img)
