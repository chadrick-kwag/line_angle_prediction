import tensorflow as tf, os, shutil, datetime

from model_builder import build_model
from dataprovider import ExistingDataProvider


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True 

# sess = tf.Session(config=config)
# tf.keras.backend.set_session(sess)





imgdirpath = "testoutput/launcher/image"
jsondirpath = "testoutput/launcher/annot"

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

# input_data_list, label_data_list = dp.get_data(2)
input_data_list, label_data_list = dp.get_all_data()


print(f"sample input_data shape: {input_data_list[0].shape}")
print(f"sample label data: {label_data_list[0]}")


test_input_data_list = input_data_list[0:2]
test_label_data_list = label_data_list[0:2]

# print(f"data list size: {len(data_list)}")
# print(data_list)



model = build_model()
model.summary()


modeljson = model.to_json()

model_arch_save_path = os.path.join(ckpt_dirpath, "model_arch.json")

with open(model_arch_save_path,'w') as fd:
    fd.write(modeljson)

print("start model compile")
# model.compile(optimizer = tf.optimizers.Adam(0.001), loss = "mse")

metric_list=[
    tf.keras.losses.MSE
]

model.compile(optimizer = tf.train.AdamOptimizer(0.0001), loss = tf.keras.losses.mse, metrics=metric_list)



checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dirpath,"weights_{epoch:02d}"),save_weights_only=True, save_best_only=True)
tfsummary_cb = tf.keras.callbacks.TensorBoard(summarydir)

callback_list=[
    checkpoint_callback,
    tfsummary_cb
]

print("start fitting...")
model.fit(input_data_list, label_data_list, epochs=100, batch_size=8, callbacks= callback_list)


pred = model.predict(test_input_data_list)

print(f"pred: {pred}")
print(f"gt: {test_label_data_list}")