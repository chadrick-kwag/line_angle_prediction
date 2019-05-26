import tensorflow as tf, os, shutil, datetime, argparse, json

from model_builder import build_model
from dataprovider import ExistingDataProvider

parser = argparse.ArgumentParser()

parser.add_argument("configfile", type=str, help="path to config file")

args = parser.parse_args()

args.configfile = args.configfile.strip()

if not os.path.exists(args.configfile):
    raise Exception(f"{args.configfile} not exist")


with open(args.configfile, 'r') as fd:
    confjson = json.load(fd)

imgdirpath = confjson["imgdirpath"]
jsondirpath = confjson["annotdirpath"]

val_imgdir = confjson["validation_imgdirpath"]
val_annotdir = confjson["validation_annotdirpath"]



restore_ckpt_path = confjson.get("restore_ckpt", None)

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
ckpt_dirpath = "ckpt/{}".format(timestamp)


summarydir = "tfsummary/{}".format(timestamp)

if not os.path.exists(ckpt_dirpath):
    os.makedirs(ckpt_dirpath)

if not os.path.exists(summarydir):
    os.makedirs(summarydir)

model_input_size = (224,224)

dp = ExistingDataProvider(imgdirpath, jsondirpath, model_input_size)
print("loading dataprovider done")

input_data_list, label_data_list = dp.get_all_data()

print(f"train data size: {len(input_data_list)}")

print(f"sample input_data shape: {input_data_list[0].shape}")
print(f"sample label data: {label_data_list[0]}")


valdp = ExistingDataProvider(val_imgdir, val_annotdir, model_input_size)

val_data = valdp.get_all_data()

print(f"validation data size: {len(val_data[0])}")


model = build_model()
# model.summary()


modeljson = model.to_json()

model_arch_save_path = os.path.join(ckpt_dirpath, "model_arch.json")

with open(model_arch_save_path,'w') as fd:
    fd.write(modeljson)

print("start model compile")
# model.compile(optimizer = tf.optimizers.Adam(0.001), loss = "mse")

metric_list=[
    tf.keras.losses.MSE
]

model.compile(optimizer = tf.train.AdamOptimizer(0.00005), loss = tf.keras.losses.mse, metrics=metric_list)



if restore_ckpt_path:
    print("start loading weights")
    model.load_weights(restore_ckpt_path)
    print(f"loading weight finished: {restore_ckpt_path}")
else:
    print("no loading weights. init from random")





checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dirpath,"weights_epoch_{epoch:04d}_vl_{val_loss:.8f}"),save_weights_only=True, save_best_only=True, monitor="val_loss")
tfsummary_cb = tf.keras.callbacks.TensorBoard(summarydir)

callback_list=[
    checkpoint_callback,
    tfsummary_cb
]

print("start fitting...")
model.fit(input_data_list, label_data_list, epochs=100, batch_size=8, callbacks= callback_list, validation_data=val_data)
