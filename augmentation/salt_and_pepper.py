import random

random.seed()

def salt_and_pepper(imgmat, distort_ratio=0.01):

    assert distort_ratio >0 and distort_ratio <1

    img_h, img_w, _ = imgmat.shape
    img_area = img_w * img_h

    attempt_size = int(img_area * distort_ratio)



    for _ in range(attempt_size):
        #pick random pixel

        random_w = random.randint(0, img_w-1)
        random_h = random.randint(0, img_h-1)

        # pick pixel color
        color = get_color_random_black_or_white()


        imgmat[random_h, random_w] = color

    return imgmat





def get_color_random_black_or_white():

    if random.random() <0.5:
        return [0,0,0]
    else:
        return [255,255,255]