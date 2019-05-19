import tensorflow as tf 


def build_model():
    model = tf.keras.applications.DenseNet201(weights=None)

    # print("model init done")
    # print(f"model: {model}")

    input_ts = model.input
    # print(f"input: {input_ts}")
    last_layer = model.layers[-2]
    # print(last_layer)

    last_layer_output = last_layer.output

    # print(f"last_layer_output: {last_layer_output}")

    fc_layer_output = tf.keras.layers.Dense(1, activation=tf.nn.relu)(last_layer_output)

    # print(f"last_layer_output: {last_layer_output}")

    final_model = tf.keras.Model(inputs=input_ts, outputs=fc_layer_output)

    # print(f"final_model: {final_model}")
    del model


    return final_model