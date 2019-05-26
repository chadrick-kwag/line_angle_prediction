import cv2, numpy as np, os

def predict_next_move_angle_rad(predictor, roi_img):

    angle_normalized = predictor.predict_single_img(roi_img)

    angle_rad = angle_normalized * 2* np.pi

    return angle_rad


def predict_for_n_runs_and_save_movement(start_coord, full_img, savedir, predictor, repeat_num):

    coord = start_coord.copy()
    full_h, full_w, _ = full_img.shape

    smaller_dim = min(full_h, full_w)
    move_unit = int(smaller_dim * 0.02)


    # draw start box
    copyimg = full_img.copy()
    p1 = tuple(coord[:2])
    p2 = tuple(coord[2:])

    cv2.rectangle(copyimg, p1,p2, (0,0,255),2)
    savepath = os.path.join(savedir, "start.png")
    cv2.imwrite(savepath, copyimg)


    for i in range(repeat_num):
        x1,y1,x2,y2 = coord
        print(f"iter:{i:02d}, coord={coord}")
        roi_img = full_img[y1:y2, x1:x2]
        angle_rad = predict_next_move_angle_rad(predictor, roi_img)

        angle_deg = angle_rad / np.pi * 180
        print(f"iter:{i:02d} , angle(deg)={angle_deg}")

        # move in the direction
        dx = move_unit * np.cos(angle_rad)
        dy = -(move_unit * np.sin(angle_rad))  # use negative since y axis direction of cv2 coord is.

        coord = shift_coord(coord, dx,dy)
        coord = [ int(x) for x in coord ] 

        # draw the new coord
        copyimg = full_img.copy()
        p1 = tuple(coord[:2])
        p2 = tuple(coord[2:])

        cv2.rectangle(copyimg, p1,p2, (0,0,255),2)
        savepath = os.path.join(savedir, f"{i:03d}.png")
        cv2.imwrite(savepath, copyimg)


def shift_coord(coord, dx,dy):
    x1,y1,x2,y2 = coord

    x1 += dx
    x2 += dx
    y1 += dy
    y2 += dy 

    return [x1,y1,x2,y2]


