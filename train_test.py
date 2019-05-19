import tensorflow as tf 

from model_builder import build_model
from dataprovider import ExistingDataProvider

imgdirpath = "testoutput/launcher_00/image"
jsondirpath = "testoutput/launcher_00/annot"

model_input_size = (224,224)

dp = ExistingDataProvider(imgdirpath, jsondirpath, model_input_size)

input_data_list, label_data_list = dp.get_data(2)

# print(f"data list size: {len(data_list)}")
# print(data_list)



model = build_model()
model.summary()

print("start model compile")
model.compile(optimizer = tf.optimizers.Adam(0.001), loss = "mse")


print("start fitting...")
model.fit(input_data_list, label_data_list, epochs=1, batch_size=2)




