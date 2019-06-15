import cv2


def low_resolution(imgmat, reduce_ratio=0.3):

    assert reduce_ratio >0 and reduce_ratio <1

    img_h, img_w, _ = imgmat.shape

    reduce_h = int(img_h * reduce_ratio)
    reduce_w = int(img_w * reduce_ratio)

    reduced_img = cv2.resize(imgmat, (reduce_w, reduce_h), interpolation=cv2.INTER_CUBIC)

    restored_img = cv2.resize(reduced_img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    return restored_img