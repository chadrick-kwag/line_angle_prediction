import json, os, tensorflow as tf , numpy as np, cv2



class Predictor:
    def __init__(self, configfile):

        assert os.path.exists(configfile)

        with open(configfile, 'r') as fd:
            confjson= json.load(fd)
        
        ckpt_path = confjson["ckpt"]
        model_input_size = confjson["model_input_size"]
        model_arch_json = confjson["model_arch_json"]
        
        self._model_input_size = tuple(model_input_size)


        # build model

        with open(model_arch_json,'r') as fd:
            modeljson = fd.read()

        model = tf.keras.models.model_from_json(modeljson)

        model.load_weights(ckpt_path)

        self._model = model
    
    def predict_batch_data(self, input_data):

        preds = self._model.predict(input_data)

        return preds

    def predict_single_img(self, img):

        img_h, img_w, _ = img.shape

        if (img_w, img_h) != self._model_input_size:
            img = cv2.resize(img, self._model_input_size)
        
        input_data = [img]
        input_data = np.array(input_data)

        preds = self.predict_batch_data(input_data)

        return preds[0]
        
