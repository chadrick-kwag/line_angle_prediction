import tensorflow as tf 

from model_builder import build_model
from dataprovider import ExistingDataProvider


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True 

# sess = tf.Session(config=config)
# tf.keras.backend.set_session(sess)





imgdirpath = "testoutput/launcher_00/image"
jsondirpath = "testoutput/launcher_00/annot"

model_input_size = (224,224)

dp = ExistingDataProvider(imgdirpath, jsondirpath, model_input_size)

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

print("start model compile")
# model.compile(optimizer = tf.optimizers.Adam(0.001), loss = "mse")
model.compile(optimizer = tf.train.AdamOptimizer(0.0001), loss = tf.keras.losses.mse)


print("start fitting...")
model.fit(input_data_list, label_data_list, epochs=100, batch_size=2)


pred = model.predict(test_input_data_list)

print(f"pred: {pred}")
print(f"gt: {test_label_data_list}")