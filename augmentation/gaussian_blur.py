import cv2

def gaussian_blur(imgmat, kernel_size=(5,5)):

    retimg = cv2.GaussianBlur(imgmat, kernel_size,0)

    return retimg

