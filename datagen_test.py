import cv2, json, numpy as np, os, shutil, random


random.seed()

img_w = 300
img_h = 300
line_width_min = 5
line_width_max = 30


line_width = random.randint(line_width_min, line_width_max)

color = (0,0,0)


filename, _ = os.path.splitext(__file__)
outputdir = f"testoutput/{filename}"

if os.path.exists(outputdir):
    shutil.rmtree(outputdir)

os.makedirs(outputdir)


blank_canvas = np.ones((img_w, img_h, 3), dtype="uint8")
blank_canvas = blank_canvas * 255

print(f"blank_canvas shape: {blank_canvas.shape}")


angle = random.random() * 2* np.pi

print(f"angle: {angle}")

cp_x_range_min = img_w * 1/4
cp_x_range_max = img_w * 3/4

cp_y_range_min = img_h * 1/4
cp_y_range_max = img_h * 3/4

cp_x = random.random() * (cp_x_range_max - cp_x_range_min) + cp_x_range_min
cp_y = random.random() * (cp_y_range_max - cp_y_range_min) + cp_y_range_min

cp_x = int(cp_x)
cp_y = int(cp_y)

print(f"centerpoint x={cp_x} y={cp_y}")
length = 500
p1_x = cp_x + length * np.cos(angle)
p1_y = cp_y + length * np.sin(angle)

p2_x = cp_x - length * np.cos(angle)
p2_y = cp_y - length * np.sin(angle)

p1_x = int(p1_x)
p2_x = int(p2_x)

p1_y = int(p1_y)
p2_y = int(p2_y)

p1= (p1_x, p1_y)
p2 = (p2_x, p2_y)
cv2.line(blank_canvas, p1,p2, color, line_width )

savepath = os.path.join(outputdir, "test.png")
cv2.imwrite(savepath, blank_canvas)


