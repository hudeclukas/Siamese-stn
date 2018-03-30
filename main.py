import tensorflow as tf
import numpy as np

from data_streamer import SUPSIM

PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/official"

def main():
    supsim = SUPSIM(PATH_2_TEXDAT)
    supsim.load_data()






if __name__ == "__main__":
    main()