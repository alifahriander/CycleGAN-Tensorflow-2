from imageio import imread, imwrite
import os 
import glob 
import shutil 
import numpy as np

PATH_CRC = "/home/ander/Desktop/CRCHistoPhenotypes_2016_04_28/Detection"
PATH_DIGIPATH = "/home/ander/PycharmProjects/nucleus_counting/data/digipath_patches_UNET_l+hxyl"

dst_crc = "./datasets/pathology/trainA"
dst_digipath = "./datasets/pathology/trainB"

dst_test_crc = "./datasets/pathology/testA"
dst_test_digipath = "./datasets/pathology/testB"

test_size = 50

nr_x = 248
nr_y = 248 
nr_c = 3

if not os.path.isdir("./datasets"):
    os.mkdir("./datasets")
    os.mkdir("./datasets/pathology")
    if  not os.path.isdir(dst_crc) and not os.path.isdir(dst_digipath):
        os.mkdir(dst_test_crc)
        os.mkdir(dst_crc)

        os.mkdir(dst_test_digipath)
        os.mkdir(dst_digipath)



# CRC Dataset 

paths_crc = glob.glob(PATH_CRC+"/*/*.bmp")

for src in paths_crc:
    print("Copying", src)
    shutil.copy(src, dst_crc)

# Paths_crc in training images 
paths_crc = glob.glob(dst_crc+"/*.bmp")

for p in paths_crc:
    im = imread(p)
    ims = []
    for i in range(2):
        for j in range(2):
            ims.append(im[nr_x*i:nr_x*(i+1), nr_y*j:nr_y*(j+1),:nr_c])
    for m, tmp in enumerate(ims):
        
        for k in range(4):
            new_path = os.path.splitext(p)[0]+"_patch"+str(m) +"_"+str(90*k) + ".jpg"
            imwrite(new_path, np.rot90(tmp,k=k))
            print("Cropped and rotated",new_path)

    os.remove(p)

paths_crc = glob.glob(dst_crc+"/*.jpg")


# Digipath Dataset 
paths_1 = np.array(glob.glob(PATH_DIGIPATH+"/*/*/ABR*.jpg"))
paths_2 = np.array(glob.glob(PATH_DIGIPATH+"/*/*/LyP*.jpg"))

subsample_1 = np.random.choice(len(paths_1), size=len(paths_crc)//2, replace=False)
subsample_2 = np.random.choice(len(paths_2), size=len(paths_crc)//2, replace=False)


paths_1 = list(paths_1[list(subsample_1)])
paths_2 = list(paths_2[list(subsample_2)])


paths_digipath = paths_1+paths_2


for p in paths_digipath:
    im = imread(p)
    # Crop 
    im = im[:nr_x, :nr_y, :nr_c]
    path = os.path.split(p)[-1]
    path = os.path.join(dst_digipath, path)
    imwrite(path, im)
    print("Cropped ", path)



    
# Create test dataset 
print("Creating test dataset with size",test_size*2)

paths_crc = np.array(glob.glob(dst_crc+"/*.jpg"))
paths_digipath = np.array(glob.glob(dst_digipath+"/*.jpg"))
test_crc = np.random.choice(len(paths_crc),size=test_size, replace=False)
test_digipath = np.random.choice(len(paths_digipath),size=test_size, replace=False)

paths_crc = list(paths_crc[list(test_crc)])
paths_digipath = list(paths_digipath[list(test_digipath)])

for i in range(len(paths_crc)):
    shutil.move(paths_crc[i], dst_test_crc)
    shutil.move(paths_digipath[i], dst_test_digipath)

