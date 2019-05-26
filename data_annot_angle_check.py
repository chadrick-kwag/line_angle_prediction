import cv2, json, os, numpy as np


annotdir = "testoutput/launcher/annot"

    
annot_files = os.listdir(annotdir)

annot_files.sort()


for f in annot_files:
    
    fullpath = os.path.join(annotdir, f)

    with open(fullpath, 'r') as fd:
        readjson = json.load(fd)

    angle = readjson["angle"]

    angle = angle / (np.pi) * 180
    
    print(f"{f} : angle={angle}")