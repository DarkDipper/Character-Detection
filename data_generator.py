import copy
import os
import xml.etree.ElementTree as et

import cv2
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from tensorflow.keras.utils import Sequence

from utils import BoundBox, bbox_iou


def parse_annotation_xml(ann_dir, img_dir, labels=None):
    if labels is None:
        labels = []
    # This parser is utilized on VOC dataset
    all_imgs = []
    seen_labels = {}

    ann_files = os.listdir(ann_dir)
    for ann in sorted(ann_files):
        img = {'object': []}

        tree = et.parse(os.path.join(ann_dir, ann))

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


def parse_annotation_csv(csv_file, labels=[], base_path=""):
    # This is a generic parser that uses CSV files
    # File_path,xmin,ymin,xmax,ymax,class

    print("parsing {} csv file can took a while, wait please.".format(csv_file))
    all_imgs = []
    seen_labels = {}

    all_imgs_indices = {}
    count_indices = 0
    with open(csv_file, "r") as annotations:
        annotations = annotations.read().split("\n")
        for i, line in enumerate(annotations):
            if line == "":
                continue
            try:
                fname, xmin, ymin, xmax, ymax, obj_name = line.strip().split(",")
                fname = os.path.join(base_path, fname)

                image = cv2.imread(fname)
                height, width, _ = image.shape

                img = dict()
                img['object'] = []
                img['filename'] = fname
                img['width'] = width
                img['height'] = height

                if obj_name == "":  # if the object has no name, this means that this image is a background image
                    all_imgs_indices[fname] = count_indices
                    all_imgs.append(img)
                    count_indices += 1
                    continue

                obj = dict()
                obj['xmin'] = int(xmin)
                obj['xmax'] = int(xmax)
                obj['ymin'] = int(ymin)
                obj['ymax'] = int(ymax)
                obj['name'] = obj_name

                if len(labels) > 0 and obj_name not in labels:
                    continue
                else:
                    img['object'].append(obj)

                if fname not in all_imgs_indices:
                    all_imgs_indices[fname] = count_indices
                    all_imgs.append(img)
                    count_indices += 1
                else:
                    all_imgs[all_imgs_indices[fname]]['object'].append(obj)

                if obj_name not in seen_labels:
                    seen_labels[obj_name] = 1
                else:
                    seen_labels[obj_name] += 1

            except:
                print("Exception occurred at line {} from {}".format(i + 1, csv_file))
                raise
    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images, config, shuffle=True, jitter=True, preprocess_input=None, callback=None):
        """

        :param images:
        :param config: must has the following keys: IMAGE_H, IMAGE_W, IMAGE_C, GRID_H, GRID_W,
        BOX (num_anchors), LABELS, CLASS (num_labels), ANCHORS, BATCH_SIZE
        :param shuffle:
        :param jitter:
        :param preprocess_input:
        :param callback:
        """

        self._images = images
        self._config = config

        self._shuffle = shuffle
        self._jitter = jitter
        self._norm = preprocess_input
        self._callback = callback

        self._anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1])
                         for i in range(int(len(config['ANCHORS']) // 2))]

        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self._aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent
                    rotate=(-5, 5),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means (kernel sizes between 2 and 7)
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians (kernel sizes between 2 and 7)
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle:
            np.random.shuffle(self._images)

    def __len__(self):
        return int(np.ceil(float(len(self._images)) / self._config['BATCH_SIZE']))

    def num_classes(self):
        return len(self._config['LABELS'])

    def size(self):
        return len(self._images)

    def load_annotation(self, i):
        annots = []

        for obj in self._images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self._config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(self._images[i]['filename'], cv2.IMREAD_GRAYSCALE)
            image = image[..., np.newaxis]
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(self._images[i]['filename'])
        else:
            raise ValueError("Invalid number of image channels.")
        return image

    def __getitem__(self, idx):
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        if r_bound > len(self._images):
            r_bound = len(self._images)
            l_bound = r_bound - self._config['BATCH_SIZE']

        instance_count = 0
        if self._config['IMAGE_C'] == 3:
            x_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'], 3))  # input images
        else:
            x_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'], 1))

        y_batch = np.zeros((r_bound - l_bound, self._config['GRID_H'], self._config['GRID_W'], self._config['BOX'],
                            4 + 1 + len(self._config['LABELS'])))  # desired network output

        for train_instance in self._images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self._jitter)

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self._config['LABELS']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self._config['IMAGE_W']) / self._config['GRID_W'])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self._config['IMAGE_H']) / self._config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self._config['GRID_W'] and grid_y < self._config['GRID_H']:
                        obj_indx = self._config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (
                                float(self._config['IMAGE_W']) / self._config['GRID_W'])
                        center_h = (obj['ymax'] - obj['ymin']) / (
                                float(self._config['IMAGE_H']) / self._config['GRID_H'])

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, center_w, center_h)

                        for i in range(len(self._anchors)):
                            anchor = self._anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

            # assign input image to x_batch
            if self._norm is not None:
                x_batch[instance_count] = self._norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[..., ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[..., ::-1], obj['name'], (obj['xmin'] + 2, obj['ymin'] + 12), 0,
                                    1.2e-3 * img.shape[0], (0, 255, 0), 2)

                x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1

        return x_batch, y_batch

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None:
            print('Cannot find ', image_name)
        if self._callback is not None:
            image, train_instance = self._callback(image, train_instance)

        h = image.shape[0]
        w = image.shape[1]
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            bbs = []
            for i, obj in enumerate(all_objs):
                xmin = obj['xmin']
                ymin = obj['ymin']
                xmax = obj['xmax']
                ymax = obj['ymax']
                # use label field to later match it with final boxes
                bbs.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax, label=i))
            bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
            image, bbs = self._aug_pipe(image=image, bounding_boxes=bbs)
            bbs = bbs.remove_out_of_image().clip_out_of_image()

            if len(bbs) < len(all_objs):
                print("Some boxes were removed during augmentations.")

            filtered_objs = []
            for bb in bbs.bounding_boxes:
                obj = all_objs[bb.label]
                obj['xmin'] = bb.x1
                obj['xmax'] = bb.x2
                obj['ymin'] = bb.y1
                obj['ymax'] = bb.y2
                filtered_objs.append(obj)
            all_objs = filtered_objs

        # resize the image to standard size
        image = cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H']))
        if self._config['IMAGE_C'] == 1:
            image = image[..., np.newaxis]
        image = image[..., ::-1]  # make it RGB (it is important for normalization of some backends)

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                obj[attr] = int(obj[attr] * float(self._config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self._config['IMAGE_H']), 0)
        return image, all_objs
