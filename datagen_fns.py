import cv2, numpy as np, random, logging



def generate_data_v1(imgsize, line_width_range, color, bg_color):
    """

    :param imgsize: tuple containing desired img size (img_w, img_h)
    :param line_width_range: tuple containing desired range of line width
    :param color: 0-255 range rgb value containing tuple (b,g,r)

    :returns: (imgmat, angle)
    """

    assert len(bg_color)==3

    img_w, img_h = imgsize
    line_width_min, line_width_max = line_width_range


    line_width = random.randint(line_width_min, line_width_max)


    blank_canvas = np.ones((img_w, img_h, 3), dtype="uint8")
    blank_canvas[:,:] = list(bg_color)

    # print(f"blank_canvas shape: {blank_canvas.shape}")


    angle = random.random() * np.pi

    cp_x_range_min = img_w * 1/4
    cp_x_range_max = img_w * 3/4

    cp_y_range_min = img_h * 1/4
    cp_y_range_max = img_h * 3/4

    cp_x = random.random() * (cp_x_range_max - cp_x_range_min) + cp_x_range_min
    cp_y = random.random() * (cp_y_range_max - cp_y_range_min) + cp_y_range_min

    cp_x = int(cp_x)
    cp_y = int(cp_y)


    length = max(img_w, img_h)

    # dx = length * np.cos(angle)
    # dy = length * np.sin(angle)

    # deg_angle = angle / np.pi * 180

    p1_x = cp_x + length * np.cos(angle)
    p1_y = cp_y - length * np.sin(angle)

    p2_x = cp_x - length * np.cos(angle)
    p2_y = cp_y + length * np.sin(angle)

    p1_x = int(p1_x)
    p2_x = int(p2_x)

    p1_y = int(p1_y)
    p2_y = int(p2_y)

    p1= (p1_x, p1_y)
    p2 = (p2_x, p2_y)

    cv2.line(blank_canvas, p1,p2, color, line_width )

    return blank_canvas, angle