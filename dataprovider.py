import os, glob, logging, random, json, cv2, numpy as np

random.seed()

logger = logging.getLogger(__name__)

class ExistingDataProvider:
    def __init__(self, imgdir, jsondir, desired_img_size):
        assert os.path.exists(imgdir)
        assert os.path.exists(jsondir)
    
        # create img_json_pathpairs
        assert isinstance(desired_img_size, tuple)
        assert len(desired_img_size) ==2
        self._desired_img_size = desired_img_size
        
        filename_to_pp_dict={}

        # base is the img dir

        imgfiles = os.listdir(imgdir)
        for f in imgfiles:
            filename, ext = os.path.splitext(f)
            if ext in [".png", ".jpg"]:

                filename_to_pp_dict[filename] = {
                    "imgpath": os.path.join(imgdir, f),
                    "jsonpath": None
                }
            
        
        # now add json files to the filename_to_pp_dict
        jsonfiles = os.listdir(jsondir)
        for f in jsonfiles:
            filename, _ = os.path.splitext(f)

            fetch_result  = filename_to_pp_dict.get(filename,None)

            if fetch_result is not None:
                filename_to_pp_dict[filename]["jsonpath"] = os.path.join(jsondir, f)
        

        # eliminate where pathpairs are not complete
        incomplete_filename_list=[]

        for k,v in filename_to_pp_dict.items():
            if v["jsonpath"] is None:
                incomplete_filename_list.append(k)

        logger.debug(f"incomplete_filename_list size: {len(incomplete_filename_list)}")

        for filename in incomplete_filename_list:
            del filename_to_pp_dict[filename]
        
        logger.debug(f"final filename_to_pp_dict size: {len(filename_to_pp_dict)}")
    
        self.filename_to_pp_dict = filename_to_pp_dict

        self._dict_key_list = list(self.filename_to_pp_dict.keys())
        self._shuffle_keys()
        self.start_index =0
    
    def get_data(self, fetch_size):
        
        assert fetch_size > 0
        start_index = self.start_index
        end_index = start_index + fetch_size

        epoch_end_signal = False
        if end_index >= len(self._dict_key_list):
            end_index = len(self._dict_key_list)
            epoch_end_signal= True
        
        selected_pp_list = []
        for i in range(start_index, end_index):
            keyname = self._dict_key_list[i]
            selected_pp_list.append(self.filename_to_pp_dict[keyname])
        
        if epoch_end_signal:
            self._shuffle_keys()

            self.start_index = 0
        else:
            self.start_index = end_index

        input_data_list=[]
        label_data_list=[]

        # grab data
        for pp in selected_pp_list:
            imgpath = pp["imgpath"]
            jsonpath = pp["jsonpath"]

            # get img
            imgmat = cv2.imread(imgpath)

            img_h, img_w, _ = imgmat.shape

            if img_h != self._desired_img_size[1] or img_w != self._desired_img_size[0]:
                resized = cv2.resize(imgmat, self._desired_img_size)
                imgmat = resized
            
            # get angle gt data

            with open(jsonpath, 'r') as fd:
                annotjson = json.load(fd)
            
            angle = annotjson["angle"]
            gt_angle = angle / np.pi
        
            input_data_list.append(imgmat)
            label_data_list.append([gt_angle])
        
        # convert to np array
        input_data_list = np.array(input_data_list)
        label_data_list = np.array(label_data_list)
        return input_data_list, label_data_list
    
    def get_all_data(self):
        input_data_list=[]
        label_data_list=[]

        for pp in self.filename_to_pp_dict.values():
            imgpath = pp["imgpath"]
            jsonpath = pp["jsonpath"]

            # get img
            imgmat = cv2.imread(imgpath)

            img_h, img_w, _ = imgmat.shape

            if img_h != self._desired_img_size[1] or img_w != self._desired_img_size[0]:
                resized = cv2.resize(imgmat, self._desired_img_size)
                imgmat = resized
            
            # get angle gt data

            with open(jsonpath, 'r') as fd:
                annotjson = json.load(fd)
            
            angle = annotjson["angle"]
            gt_angle = angle / np.pi
        
            input_data_list.append(imgmat)
            label_data_list.append([gt_angle])
        
        input_data_list = np.array(input_data_list)
        label_data_list = np.array(label_data_list)

        return input_data_list, label_data_list

    def _shuffle_keys(self):
        random.shuffle(self._dict_key_list)




