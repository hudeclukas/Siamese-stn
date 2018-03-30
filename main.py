import tensorflow as tf
import numpy as np
import os
import time

from data_streamer import SUPSIM

PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/official"
MAX_ITERS = 25001
MODEL_NAME = "siam_stn_001"

def main():
    supsim = SUPSIM(PATH_2_TEXDAT)
    print("Starting loading data")
    start = time.time() * 1000
    supsim.load_data()
    print((time.time() * 1000) - start, "ms")







if __name__ == "__main__":
    main()