import os, shutil, argparse
from tqdm import tqdm
from multiprocessing import Pool
from work_thread import gen_data_and_save
from augmentation.augmentor import Augmentor


parser = argparse.ArgumentParser()

parser.add_argument("gensize",type=int,help="size of data to generate")
parser.add_argument("augmentconfig", type=str, help="augmentor config file path")

args = parser.parse_args()

if args.gensize <=0:
    raise Exception("gensize should be > 0")

args.augmentconfig = args.augmentconfig.strip()

assert os.path.exists(args.augmentconfig)



augmentor = Augmentor(args.augmentconfig)

print("augmentor config: {}".format(augmentor.conf))



filename, _ = os.path.splitext(__file__)
outputdir = f"testoutput/{filename}"

if os.path.exists(outputdir):
    shutil.rmtree(outputdir)

os.makedirs(outputdir)


img_savedir = os.path.join(outputdir, "image")
json_savedir = os.path.join(outputdir, "annot")

os.makedirs(img_savedir)
os.makedirs(json_savedir)


imgsize = (300,300)
line_width_range = (5,30)
color = (0,0,0)
bgcolor = (255,255,255)

pool = Pool(processes=3)

gen_image_num = args.gensize

arg_list=[]
async_result_list=[]

for i in tqdm(range(gen_image_num)):
    arg = (imgsize, line_width_range, color, bgcolor, augmentor, f"{i:08d}", img_savedir, json_savedir)
    # arg_list.append(arg)\
    result = pool.apply_async(gen_data_and_save, arg )
    async_result_list.append(result)


for i,res in enumerate(async_result_list):
    res.get()
    if i%100==0:
        print("{} done".format(i))

