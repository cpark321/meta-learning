import os
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

def load(data_type, num_classes):
    datadir = "/home/mifs/pm574/MLMI4/data/miniImagenet/{}/".format(data_type)
    classes = os.listdir(datadir)
    print(len(classes))
    import pdb; pdb.set_trace()
    X = np.zeros((num_classes, 600, 84, 84, 3), dtype=np.uint8)

    for ci, c in tqdm(enumerate(classes)):
        class_dir = datadir + c
        fnames = os.listdir(class_dir)
        for ei, fname in enumerate(fnames):
            fpath = class_dir + '/' + fname
            image = np.array(Image.open(fpath), dtype=np.uint8)
            X[ci, ei,:,:,:] = image

    import pdb; pdb.set_trace()

    savepath = "/home/mifs/pm574/MLMI4/data/miniimagenet.{}.nparray.pk".format(data_type)
    with open(savepath, "wb") as f:
        pickle.dump(X, f)

load('test', num_classes=20)
