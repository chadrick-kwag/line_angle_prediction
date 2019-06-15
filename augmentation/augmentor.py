import os, json, random
from .salt_and_pepper import salt_and_pepper
from .gaussian_blur import gaussian_blur
from .low_resolution import low_resolution

random.seed()

class Augmentor:
    def __init__(self, confjsonfile):

        self.conf = AugmentorConfig(confjsonfile)

    def apply(self, imgmat):


        chance = random.random()

        if chance < self.conf.salt_and_pepper_prob:
            imgmat = salt_and_pepper(imgmat)

        chance = random.random()

        if chance < self.conf.gaussian_blur_prob:
            imgmat = gaussian_blur(imgmat, kernel_size=(11,11))

        chance = random.random()

        if chance < self.conf.low_resolution_prob:
            imgmat = low_resolution(imgmat)


        return imgmat





class AugmentorConfig:
    def __init__(self, jsonfile):

        assert os.path.exists(jsonfile)

        with open(jsonfile, 'r') as fd:
            confjson = json.load(fd)


        self.salt_and_pepper_prob = confjson.get("salt_and_pepper_prob", 0)
        self.gaussian_blur_prob = confjson.get("gaussian_blur_prob",0)
        self.jpeg_compression_loss_prob = confjson.get("jpeg_compression_loss_prob",0)
        self.low_resolution_prob = confjson.get("low_resolution_prob",0)


    def __repr__(self):

        summary_dict={
            "salt_and_pepper_prob": self.salt_and_pepper_prob,
            "gaussian_blur_prob": self.gaussian_blur_prob,
            "jpeg_compression_loss_prob": self.jpeg_compression_loss_prob,
            "low_resolution_prob": self.low_resolution_prob
        }

        return str(summary_dict)


