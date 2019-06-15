import unittest, cv2, os
from salt_and_pepper import salt_and_pepper
from gaussian_blur import gaussian_blur
from low_resolution import low_resolution




class test1(unittest.TestCase):

    def setUp(self):
        imgpath = "testinput/00000000.png"
        self.img = cv2.imread(imgpath)
        self.outputdir = "testoutput"

        if not os.path.exists(self.outputdir):

            os.makedirs(self.outputdir)



    def test_salt_and_pepper(self):
        img = self.img.copy()
        retimg = salt_and_pepper(img)

        savepath = os.path.join(self.outputdir, "test_salt_and_pepper.png")
        cv2.imwrite(savepath, retimg)


    def test_gaussian_blur(self):
        img = self.img.copy()

        retimg = gaussian_blur(img, kernel_size=(11,11))
        savepath = os.path.join(self.outputdir, "test_gaussian_blur.png")
        cv2.imwrite(savepath, retimg)

    def test_low_resolution(self):

        img = self.img.copy()
        retimg = low_resolution(img, reduce_ratio=0.2)

        savepath = os.path.join(self.outputdir, "test_low_resolution.png")
        cv2.imwrite(savepath, retimg)



if __name__ == "__main__":
    unittest.main()




