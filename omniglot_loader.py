import os
import sys
import pdb
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

class OmniglotDataLoader(object):
    def __init__(self):
        datapath = "/home/mifs/pm574/MLMI4/data/omniglot_resized/"
        alphabets = os.listdir(datapath)
        idx = 0
        X_data = np.zeros((1623, 20, 28, 28), dtype=np.uint8)
        # 1623 = num_characters
        # 20   = each character has 20 instances
        for alphabet in tqdm(alphabets):
            ap_dir = datapath + alphabet
            characters = os.listdir(ap_dir)
            for character in characters:
                char_dir = ap_dir + "/" + character
                for j, fname in enumerate(os.listdir(char_dir)):
                    fpath = char_dir + "/" + fname
                    image = np.array(Image.open(fpath), dtype=np.uint8)
                    X_data[idx, j, :, :] = image
                    # print("idx = {} --- j = {}".format(idx, j))
                idx += 1
        with open("/home/mifs/pm574/MLMI4/data/omniglot.nparray.pk", "wb") as f:
            pickle.dump(X_data, f)

loader = OmniglotDataLoader()
