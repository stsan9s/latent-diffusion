import os
import json

from taming.data.base import ImagePaths
from taming.data.faceshq import FacesBase

class ImagePathsFFHQLabels(ImagePaths):
    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            if k == 'class_label' or k == 'human_label':
                file_num = self.labels['file_path_'][i].split('/')[-1][:-4]
                example[k] = self.labels[k][file_num]
            else:
                example[k] = self.labels[k][i]
        return example


class FFHQTrainLabels(FacesBase):
    def __init__(self, size, keys=None, root="data/ffhq"):
        super().__init__()
        if root[0] == '$':
            root = os.environ[root[1:]]
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]

        with open('data/ffhq-features/emotion_numeric_class.json', 'r') as f:
            labels = {'class_label': json.load(f)}
        with open('data/ffhq-features/emotion_dict.json', 'r') as f:
            labels['human_label'] = json.load(f)

        self.data = ImagePathsFFHQLabels(paths=paths, size=size, random_crop=False, labels=labels)
        self.keys = keys


class FFHQValidationLabels(FacesBase):
    def __init__(self, size, keys=None, root="data/ffhq"):
        super().__init__()
        if root[0] == '$':
            root = os.environ[root[1:]]
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]

        with open('data/ffhq-features/emotion_numeric_class.json', 'r') as f:
            labels = {'class_label': json.load(f)}
        with open('data/ffhq-features/emotion_dict.json', 'r') as f:
            labels['human_label'] = json.load(f)

        self.data = ImagePathsFFHQLabels(paths=paths, size=size, random_crop=False, labels=labels)
        self.keys = keys

class FFHQTestLabels(FacesBase):
    def __init__(self, size, keys=None, root="data/ffhq"):
        super().__init__()
        if root[0] == '$':
            root = os.environ[root[1:]]
        with open("data/ffhqtest.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]

        with open('data/ffhq-features/emotion_numeric_class.json', 'r') as f:
            labels = {'class_label': json.load(f)}
        with open('data/ffhq-features/emotion_dict.json', 'r') as f:
            labels['human_label'] = json.load(f)

        self.data = ImagePathsFFHQLabels(paths=paths, size=size, random_crop=False, labels=labels)
        self.keys = keys