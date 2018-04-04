import tensorflow as tf
from tensorflow.contrib import slim
from math import sqrt

def image_summary(var: tf.Tensor, image_size: tuple, index: int, name: str):
    iyr = int(450 / image_size[0])
    ixr = int(450 / image_size[1])
    ix = image_size[0]
    iy = image_size[1]
    sliced = tf.slice(var, (index, 0, 0, 0), (1, -1, -1, -1))
    sliced = tf.image.resize_images(
        images=sliced,
        size=(iyr * iy, ixr * ix),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    slim.summary.image(name, sliced)

def conv_image_summary(conv_layer:tf.Tensor, patch_size:tuple, index:int, name:str):
        iyr = int(650 / patch_size[0])
        ixr = int(650 / patch_size[1])
        ix = patch_size[0]
        iy = patch_size[1]
        i_shape = round(sqrt(patch_size[2]))
        image = tf.slice(conv_layer, (index, 0, 0, 0), (1, -1, -1, -1))
        image = tf.reshape(image, patch_size)
        ix += 2
        iy += 2
        image = tf.image.resize_image_with_crop_or_pad(image, iy, ix)
        image = tf.reshape(image, (iy, ix, i_shape, i_shape))
        image = tf.transpose(image, (2, 0, 3, 1))
        image = tf.reshape(image, (1, i_shape * iy, i_shape * ix, 1))
        image = tf.image.resize_images(
            images=image,
            size=(iyr * iy, ixr * ix),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        tf.summary.image(name, image)

def print_layer(layer:tf.Tensor):
    values = layer.eval()
    print('Layer: {:s}'.format(layer.name))
    print(values)