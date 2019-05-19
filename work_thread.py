from datagen_fns import generate_data_v1
import cv2, json, os

def gen_data_and_save(imgsize, line_width_range, color, savename, img_savedir, json_savedir):

    imgmat, angle = generate_data_v1(imgsize, line_width_range, color)

    img_savepath = os.path.join(img_savedir, "{}.png".format(savename))
    cv2.imwrite(img_savepath, imgmat)

    json_savepath = os.path.join(json_savedir, "{}.json".format(savename))

    savejson={
        "angle": angle
    }

    with open(json_savepath, 'w') as fd:
        json.dump(savejson, fd)
    




