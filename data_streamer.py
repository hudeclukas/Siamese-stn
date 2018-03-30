import numpy as np
import os
import random
import matplotlib.pyplot as plt

from array import array
from skimage import transform
import sklearn.preprocessing as prep

def resize_batch_images(batch: list, image_size: tuple) -> np.ndarray:
    return np.asarray([transform.resize(image, image_size, mode="reflect") for image in batch])

class ImageObjects:
    def __init__(self):
        self.objects = []
        self.name = ''

    def write_similarities_to_file(self, path: str, labels_order: list, ith: int):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        with open(path, mode='ab+') as bf:
            if ith == 0:
                bf.write(len(labels_order).to_bytes(4, 'little', signed=False))
                lb = array('i', labels_order)
                lb.tofile(bf)
            if ith >= 0:
                bf.write(len(self.objects[ith].labels).to_bytes(4, 'little', signed=False))
                for j in range(len(self.objects[ith].labels)):
                    sim = array('d', self.objects[ith].similarity[j])
                    sim.tofile(bf)

            bf.close()
        print('File saved {:s}'.format(path))

def vizualize_batch(batch_1, batch_2):
    viz = [np.concatenate((sup1, sup2), 1) for sup1, sup2 in zip(batch_1, batch_2)]
    viz = np.vstack(viz)
    viz = viz.transpose([1, 0, 2])
    plt.imshow(viz)


class ObjectSuperpixels:
    def __init__(self):
        self.superpixels = []
        self.labels = []
        self.similarity = []


class SUPSIM:

    def __init__(self, path):
        if not os.path.exists(os.path.abspath(path)):
            raise Exception('Path does not exists')
        self.train = SUPSIM.train(os.path.abspath(path+'\\train'))
        self.test = SUPSIM.test(os.path.abspath(path+'\\test'))

    def load_data(self):
        self.train._load_data()
        self.test._load_data()


    class train:
        class teacher:
            def __init__(self):
                # pairs
                self.fp = []
                # singles
                self.fn = []

        def __init__(self, path):
            self.path = path
            self.images = []
            self.data = []
            self.teacher = self.teacher()
            self.streamer = SUPSIM._data_streamer()

        def _load_data(self):
            self.data, self.images = self.streamer.read_data_to_array(self.path)

        def next_batch(self, batch_size:int, with_indexes:bool):
            return self.streamer.random_batch(self.data, batch_size, return_classes=with_indexes)

        def next_teacher_batch(self, batch_size):
            batch_size_teacher = int(0.2 * batch_size)
            fn_size = batch_size_teacher >> 1
            fn_size = fn_size if fn_size < len(self.teacher.fn) else len(self.teacher.fn)
            fp_size = batch_size_teacher - fn_size
            fp_size = fp_size if fp_size < len(self.teacher.fp) else len(self.teacher.fp)
            batch_size_classic = batch_size - (fp_size + fn_size)
            batch_1, batch_2, labels = self.streamer.random_batch(self.data, batch_size_classic)
            if len(self.teacher.fp) > 0:
                fp = np.array(self.teacher.fp)
                batch_fp_idx = np.random.choice(np.arange(fp.shape[0]), fp_size, replace=False)
                batch_fp_idx = fp[batch_fp_idx]
                batch_fp_idx = batch_fp_idx.transpose()
                batch_fp_classes_1 = np.array(self.data)[batch_fp_idx[0]]
                batch_fp_classes_2 = np.array(self.data)[batch_fp_idx[1]]
                batch_fp_1 = np.array([random.choice(sp.superpixels) for sp in batch_fp_classes_1])
                batch_fp_2 = np.array([random.choice(sp.superpixels) for sp in batch_fp_classes_2])
                batch_1 = np.concatenate((batch_1, batch_fp_1))
                batch_2 = np.concatenate((batch_2, batch_fp_2))
                labels = np.concatenate((labels, np.zeros(fp_size, dtype=np.float32)))
            if len(self.teacher.fn) > 0:
                fn = self.teacher.fn
                batch_fn_idx = random.sample(fn, fn_size)
                batch_fn_classes = np.array(self.data)[batch_fn_idx]
                batch_fn = np.array([random.sample(sp.superpixels, 2) for sp in batch_fn_classes])
                axes = np.arange(len(batch_fn.shape))
                a, b, *c = axes
                axes = np.concatenate((np.array((b, a), dtype=int), c)).astype(int)
                batch_fn = batch_fn.transpose(axes)
                batch_1 = np.concatenate((batch_1, batch_fn[0]))
                batch_2 = np.concatenate((batch_2, batch_fn[1]))
                labels = np.concatenate((labels, np.ones(fn_size, dtype=np.float32)))

            return batch_1, batch_2, labels

    class test:
        def __init__(self, path):
            self.path = path
            self.images = []
            self.data = []
            self.streamer = SUPSIM._data_streamer()

        def _load_data(self):
            self.data, self.images = self.streamer.read_data_to_array(self.path)

        def next_batch(self, batch_size:int, with_indexes:bool):
            return self.streamer.random_batch(self.data, batch_size, return_classes=with_indexes)


    class _data_streamer:

        def read_data_to_array(self, abspath):
            if not os.path.exists(abspath):
                raise IOError('Attempt to read_read_data_to_array. Path not found!')

            print("Loading from \"" + abspath + "\"")
            out_data = []
            out_images = []
            files = os.listdir(abspath)
            for file in files:
                img_objs = ImageObjects()
                img_objs.name = file
                try:
                    with open(os.path.join(abspath, file), "rb") as bf:
                        channels = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        objs = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        for o in range(objs):
                            obj_sups = ObjectSuperpixels()
                            sups = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                            for s in range(sups):
                                if file.endswith('.supl'):
                                    label = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                    obj_sups.labels.append(label)
                                rows = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                cols = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                data = np.empty(shape=[rows, cols, channels], dtype=np.ubyte)
                                bf.readinto(data.data)
                                data = prep.scale(data.reshape((rows * cols * channels))).reshape(
                                    (rows, cols, channels))
                                obj_sups.superpixels.append(data)
                            out_data.append(obj_sups)
                            img_objs.objects.append(obj_sups)
                        bf.close()
                    out_images.append(img_objs)
                except IOError:
                    print("File {:s} does not exist".format(os.path.join(abspath, file)))
            print(str(len(out_data)) + " objects loaded")
            return np.array(out_data), np.array(out_images)

        def random_batch(self, data, batch_size, image_size=None, return_classes=False):
            neg_size = batch_size
            pos_size = batch_size >> 1
            neg_classes_idx = np.random.choice(np.arange(len(data) - 1), neg_size, False)
            pos_classes_idx = np.random.choice(np.arange(len(data) - 1), pos_size, False)
            pos_classes = data[pos_classes_idx]
            neg_classes = data[neg_classes_idx]
            neg_pairs_count = int(neg_size) >> 1
            neg_s = [random.choice(c.superpixels) for c in neg_classes]
            neg_s_1 = neg_s[0:neg_pairs_count]
            neg_s_2 = neg_s[neg_pairs_count:neg_size]
            neg_l = np.zeros(neg_pairs_count, dtype=np.float32)
            pos_s = []
            for i in range(pos_size):
                pos_s.append(random.sample(pos_classes[i].superpixels, 2))
            pos_s = np.array(pos_s)
            axes = np.arange(len(pos_s.shape))
            a, b, *c = axes
            axes = np.concatenate((np.array((b, a), dtype=int), c)).astype(int)
            pos_s = np.array(pos_s).transpose(axes)
            pos_s_1 = pos_s[0]
            pos_s_2 = pos_s[1]
            pos_l = np.ones(pos_size, dtype=np.float32)
            batch_s_t_1 = np.concatenate((neg_s_1, pos_s_1))
            batch_s_t_2 = np.concatenate((neg_s_2, pos_s_2))
            batch_l_t = np.concatenate((neg_l, pos_l))

            if return_classes:
                neg_s_idx = neg_classes_idx.reshape((neg_pairs_count, 2), order='F')
                pos_s_idx = np.concatenate((pos_classes_idx.reshape((pos_size, 1)), pos_classes_idx.reshape((pos_size, 1))),
                                           axis=1)
                idx = np.concatenate((neg_s_idx, pos_s_idx))
                return batch_s_t_1, batch_s_t_2, batch_l_t, idx
            else:
                return batch_s_t_1, batch_s_t_2, batch_l_t
