import cv2, os, json, shutil
import tensorflow as tf , numpy as np
from dataprovider import ExistingDataProvider



model_json_path = "ckpt/190527_000322/model_arch.json"
weight_path = "ckpt/190527_000322/weights_epoch_0026_vl_0.00002115"

# testimg = "/home/chadrick/prj/line_angle_prediction/testoutput/datagen_test/test.png"
loadimgdir = "testoutput/quickrun_train_data/image"
loadannotdir = "testoutput/quickrun_train_data/annot"


outputdir = "testoutput/predict_1"

if os.path.exists(outputdir):
    shutil.rmtree(outputdir)
os.makedirs(outputdir)

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

for i, (img, pred_val, gt_val) in enumerate(zip(input_data_list, pred_result, label_data_list)):
    savepath = os.path.join(outputdir, f"{i:03d}.png")
    cv2.imwrite(savepath, img)

    angle = pred_val * 180
    print(f"index: {i:03d}, pred_val: {pred_val}, gt_val={gt_val}, pred_angle(deg): {angle}")

