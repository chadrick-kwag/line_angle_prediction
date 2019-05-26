import cv2, os, json
import tensorflow as tf , numpy as np
from dataprovider import ExistingDataProvider



model_json_path = "ckpt/190525_161854/model_arch.json"
weight_path = "ckpt/190525_161854/weights_34"

# testimg = "/home/chadrick/prj/line_angle_prediction/testoutput/datagen_test/test.png"
loadimgdir = "testoutput/launcher_00/image"
loadannotdir = "testoutput/launcher_00/annot"

model_input_size = (224,224)

dp = ExistingDataProvider(loadimgdir, loadannotdir, model_input_size)

input_data_list, label_data_list = dp.get_all_data()

print(f"data size: {len(input_data_list)}")

assert os.path.exists(model_json_path)

with open(model_json_path,'r') as fd:
    modeljson = fd.read()

model = tf.keras.models.model_from_json(modeljson)

model.load_weights(weight_path)

pred_result = model.predict(input_data_list)

print(f"pred_result : {pred_result}")

for pred_val, gt_val in zip(pred_result, label_data_list):
    angle = pred_val * 360
    print(f"pred_val: {pred_val}, gt_val={gt_val}, pred_angle: {angle}")

