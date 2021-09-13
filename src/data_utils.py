import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torchvision
import cv2

from . import transforms

from bs4 import BeautifulSoup
import json


andrewmvd_classes = {
    'correct': ['with_mask', ],
    'incorrect': ['mask_weared_incorrect', ],
    'no_mask': ['without_mask', ],
}

wobotintelligence_classes = {
    'correct': ['face_with_mask', ],
    'incorrect': ['face_with_mask_incorrect', ],
    'no_mask': ['face_no_mask', 'face_other_covering', ],
}


def generate_label(name, correct, incorrect, no_mask):
    if name in correct:
        return 3
    elif name in incorrect:
        return 2
    elif name in no_mask:
        return 1
    return 0


def xml_generator(path):
    with open(path) as f:
        for annotation in BeautifulSoup(f.read(), 'lxml').find_all('object'):
            label = annotation.find('name').text
            box = [int(annotation.find('xmin').text), int(annotation.find('ymin').text), int(annotation.find('xmax').text), int(annotation.find('ymax').text)]
            yield box, label


def json_generator(path):
    with open(path) as f:
        for annotation in json.load(f)['Annotations']:
            label = annotation['classname']
            box = annotation['BoundingBox']
            yield box, label


def generate_target(path, generator, classes, process_labels=True):
    boxes = []
    labels = []
    for box, label in generator(path):
        if process_labels:
            label = generate_label(label, correct=classes['correct'], incorrect=classes['incorrect'], no_mask=classes['no_mask'])
            if label == 0:
                continue
        
        if box[0] < box[2] and box[1] < box[3]:
            labels.append(label)
            boxes.append(box)
        
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if process_labels:
        labels = torch.as_tensor(labels, dtype=torch.int64)

    

    if len(boxes.shape) < 2:
        boxes = torch.zeros((0, 4),dtype=torch.float32)
    return {'boxes': boxes, 'labels': labels}


def collate_fn(batch):
    return list(zip(*batch))


def train_test_split(cls, dataset, p=0.25, seed=42):
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(dataset.items)
    test_size = int(len(dataset.items) * p)

    train = cls(dataset.path, transforms=dataset.transforms)
    train.items = dataset.items[test_size:]
    
    test = cls(dataset.path, transforms=dataset.transforms)
    test.items = dataset.items[:test_size]
    return train, test


def count_labels(dataset, process_labels=True, tqdm_bar=False):
    labels = pd.DataFrame(dataset.items, columns=['item', 'dataset'])
    labels_set = set()
    if tqdm_bar:
        bar = tqdm(range(len(dataset)))
    else:
        bar = range(len(dataset))

    for i in bar:
        target = dataset.load_annotation(i, process_labels=process_labels)
        label = target['labels'].tolist() if process_labels else target['labels']
        label = pd.Series(label, dtype=int).value_counts()
        labels.loc[i, label.index] = label.values
        labels_set.update(label.index)

    labels.loc[:, labels_set] = labels.loc[:, labels_set].fillna(0)
    label_counts = {l: int(labels[l].sum()) for l in sorted(labels_set)}
    label_counts = dict(sorted(label_counts.items(), key=lambda x: x[0], reverse=True))
    
    label_weights = labels[label_counts.keys()].copy()
    label_weights['count'] = labels[label_counts.keys()].sum(axis=1)
    for col in label_counts.keys():
        label_weights['w' + str(col)] = (label_weights[col] / label_counts[col]) / label_weights['count']
    label_weights['weight'] = label_weights[['w' + str(col) for col in label_counts.keys()]].sum(axis=1)

    return label_counts, label_weights

class UnionDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, dims=(320, 320)):
        super().__init__()
        self.dims = dims
        self.datasets = datasets
        self.items = []
        for i, dataset in enumerate(self.datasets):
            for item in range(len(dataset)):
                self.items.append((item, i))

    def get_inverse_weights(self, tqdm_bar=False):
        self.label_counts, self.label_weights = count_labels(self, process_labels=True, tqdm_bar=tqdm_bar)
        self.item_weights = self.label_weights['weight'].values

    def load_annotation(self, i, process_labels=True):
        i, dataset = self.items[i]
        dataset = self.datasets[dataset]
        target = dataset.load_annotation(i, process_labels=process_labels)
        return target

    def resize(self, image, boxes):
        new_image = torchvision.transforms.functional.resize(image, self.dims)

        # Resize bounding boxes
        old_dims = torch.FloatTensor([image.shape[2], image.shape[1], image.shape[2], image.shape[1]]).unsqueeze(0)
        new_boxes = boxes.clone() / old_dims  # percent coordinates

        new_dims = torch.FloatTensor([self.dims[1], self.dims[0], self.dims[1], self.dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

        return new_image, new_boxes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item, dataset = self.items[i]
        img, target = self.datasets[dataset][item]
        
        img, target['boxes'] = self.resize(img, target['boxes'])
        
        return img, target


class AndrewmvdDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transforms: torchvision.transforms.Compose=None):
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.items = sorted([int(i[12:][:-4]) for i in os.listdir(os.path.join(self.path, 'annotations'))])

    def load_image(self, i):
        i = self.items[i]
        return torch.tensor(cv2.imread(os.path.join(self.path, 'images', f'maksssksksss{i}.png'))[:, :, ::-1].copy()).float().permute(2, 0, 1)

    def load_annotation(self, i, process_labels=True):
        i = self.items[i]
        return generate_target(os.path.join(self.path, 'annotations', f'maksssksksss{i}.xml'),
                               generator=xml_generator,
                               classes=andrewmvd_classes,
                               process_labels=process_labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.load_image(i)
        target = self.load_annotation(i, process_labels=True)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class WobotIntelligenceDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transforms: torchvision.transforms.Compose=None):
        super().__init__()
        self.path = os.path.join(path, 'Medical mask', 'Medical mask', 'Medical Mask')
        self.transforms = transforms
        self.items = [i[:-5] for i in sorted(os.listdir(os.path.join(self.path, 'annotations')))]

    def load_image(self, i):
        i = self.items[i]
        return torch.tensor(cv2.imread(os.path.join(self.path, 'images', i))[:, :, ::-1].copy()).float().permute(2, 0, 1)

    def load_annotation(self, i, process_labels=True):
        i = self.items[i]
        return generate_target(os.path.join(self.path, 'annotations', f'{i}.json'),
                               generator=json_generator,
                               classes=wobotintelligence_classes,
                               process_labels=process_labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.load_image(i)
        target = self.load_annotation(i, process_labels=True)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


train_transforms = transforms.Compose(transforms=[
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomIoUCrop(),
    transforms.RandomZoomOut(),
],
    img_transforms=[
    torchvision.transforms.RandomGrayscale(p=0.2),
])

