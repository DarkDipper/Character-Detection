import sys
import numpy as np
import argparse
import random
import json

from data_generator import parse_annotation_xml

num_anchors = 5
config = {
    'model': {
        'input_size_h': 400,
        'input_size_w': 400,
        'output_shape_w': 11,
        'output_shape_h': 11,
        'labels': [
          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
          'U', 'V', 'W', 'X', 'Y', 'Z']
    },
    'train': {
        'train_annotation_folder': '/content/train/anns/',
        'train_image_folder': '/content/train/images/'
    }

}


def iou(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_iou(anns, centroids):
    n, d = anns.shape
    _sum = 0.
    for i in range(anns.shape[0]):
        _sum += max(iou(anns[i], centroids))

    return _sum/n


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.5f,%0.5f, ' % (anchors[i, 0], anchors[i, 1])

    # there should not be comma after last anchor, that's why
    r += '%0.5f,%0.5f' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1])
    r += "]"

    print(r)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for _ in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - iou(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)  # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def main():
    train_imgs, train_labels = parse_annotation_xml(config['train']['train_annotation_folder'],
                                                    config['train']['train_image_folder'],
                                                    config['model']['labels'])

    input_size = (config['model']['input_size_h'], config['model']['input_size_w'], 3)
    grid_w = config['model']['input_size_w'] / config['model']['output_shape_w']
    grid_h = config['model']['input_size_h'] / config['model']['output_shape_h']

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        cell_w = image['width']/grid_w
        cell_h = image['height']/grid_h

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin']))/cell_w
            relative_h = (float(obj["ymax"]) - float(obj['ymin']))/cell_h
            annotation_dims.append(tuple(map(float, (relative_w, relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_iou(annotation_dims, centroids))
    print_anchors(centroids)


if __name__ == '__main__':
    main()