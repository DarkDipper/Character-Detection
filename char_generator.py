from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np


class CharGenerator(Sequence):
    def __init__(self, batch_size, images, label_map):
        self.batch_size = batch_size
        self.images = images
        self.label_map = label_map
        self.indices = np.random.permutation(len(self.images))

    def __len__(self):
        return int(len(self.images)/self.batch_size)

    def __getitem__(self, index):
        list_char_images = []
        list_char_labels = []
        for idx_batch in range(self.batch_size):
            idx_image = self.indices[index*self.batch_size+idx_batch]
            image_info = self.images[idx_image]
            np_image = np.array(Image.open(image_info['filename']))
            num_chars = len(image_info['object'])
            idx_char = np.random.randint(16)
            if idx_char < num_chars:
                char_info = image_info['object'][idx_char]
                label = char_info['name']
                column = int(char_info['xmin']/160)
                row = int(char_info['ymin']/160)
            else:
                label = '?'
                row = int(idx_char/4)
                column = idx_char % 4

            char_image = np_image[row *
                                  160:(row+1)*160, column*160:(column+1)*160, :]
            list_char_images.append(char_image)
            list_char_labels.append(self.label_map[label])
        return np.array(list_char_images), np.array(list_char_labels)

    def on_end_epoch(self):
        self.indices = np.random.permutation(len(self.images))
