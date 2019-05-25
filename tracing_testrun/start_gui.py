import cv2


def fetch_start_and_endpoint():
    sp = central_data["startpoint"]
    ep = central_data["endpoint"]

    return list(sp), list(ep)

def move_coord_by_dx_dy(coord, dx,dy):
    x,y = coord
    x = x + dx
    y = y + dy
    return [x,y]

def update_start_and_endpoint(sp,ep):
    central_data["startpoint"] = tuple(sp)
    central_data["endpoint"] = tuple(ep)


testimg = "../testinput/track_run_testimg/track_testimg_0.png"


central_data={
    "startpoint": None,
    "endpoint": None
}

def callback(event, x,y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        central_data["startpoint"] = x,y
        print(f"setting startpoint={(x,y)}")
    
    elif event == cv2.EVENT_LBUTTONUP:
        central_data["endpoint"] = x,y
        print(f"setting endpoint = {(x,y)}")

    
    
# up: 82
# down: 84
# left: 81
# right: 83


direction_key_val = [81,82,83,84,119,97, 100, 115]
direction_map={
    81: "left",
    82: "up",
    83: "right",
    84: "down",
    119: "up",
    97: "left",
    100: "right",
    115: "down"
}

confirm_key_val = [13]

move_precision = 0.01

img = cv2.imread(testimg)
central_data["img"] = img
cv2.namedWindow("image")
cv2.setMouseCallback("image", callback)

confirm_flag = False

while True:

    draw_img = img.copy()
    img_h, img_w, _ = draw_img.shape
    points_exist_flag = False
    if central_data["startpoint"] is not None and central_data["endpoint"] is not None:
        points_exist_flag = True
        p1 = central_data["startpoint"]
        p2 = central_data["endpoint"]

        cv2.rectangle(draw_img, p1,p2,(0,0,255),2)
    print(f"points_exist_flag: {points_exist_flag}")

    cv2.imshow("image", draw_img)
    retkey = cv2.waitKey(25)

    shift_x = img_w * move_precision
    shift_y = img_h * move_precision

    shift_x = int(shift_x)
    shift_y = int(shift_y)

    if retkey != -1:
        print(retkey)

    if retkey in direction_key_val :

        print("direction key detected. points_exist_flag={}".format(points_exist_flag))

        if points_exist_flag:


            direction = direction_map[retkey]

            sp,ep = fetch_start_and_endpoint()
            if direction =="up":
                sp = move_coord_by_dx_dy(sp, 0, -shift_y)
                ep = move_coord_by_dx_dy(ep, 0, -shift_y)

            elif direction == "down":
                sp = move_coord_by_dx_dy(sp, 0, shift_y)
                ep = move_coord_by_dx_dy(ep, 0, shift_y)
            elif direction == "right":
                sp = move_coord_by_dx_dy(sp, shift_x, 0)
                ep = move_coord_by_dx_dy(ep, shift_x, 0)
            elif direction == "left":
                sp = move_coord_by_dx_dy(sp, -shift_x, 0)
                ep = move_coord_by_dx_dy(ep, -shift_x, 0)
            else:
                raise Exception(f"invalid direction={direction}")
                

            update_start_and_endpoint(sp,ep)
        
        continue
    
    if retkey in confirm_key_val:
        confirm_flag=True

    if retkey != -1 or retkey == ord('q') or confirm_flag:
        print(retkey)
        break

cv2.destroyAllWindows()


print(f"confirm flag = {confirm_flag}")