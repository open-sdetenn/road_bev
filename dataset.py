import torch
from torch.utils.data import Dataset
import numpy as np

def img_to_categorical(img, needed_labels):
  
  cat = np.empty((img.shape[0], img.shape[1], len(needed_labels)))

  for channel, label in enumerate(needed_labels):
    cat[:, :, channel] = np.where(np.isin(img, label), 1, 0)

  return cat

def categorical_to_img(cat):
  
  img = np.argmax(cat, axis=-1)
  return img

needed_labels = [0, 1, 6, 7, 8, 10, 11]

class Dataset(Dataset):
    def __init__(self, img_paths, base_path='', to_fit=True, batch_size=32, shuffle=True, needed_classes=[]):
        self.img_paths = img_paths.copy()
        self.base_path = base_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.needed_classes = needed_classes
        self.on_epoch_end()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        X = self._generate_X(img_path)

        if self.to_fit:
            y = self._generate_y(img_path)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            import random
            random.shuffle(self.img_paths)

    def _generate_X(self, img_path):
        img = self._load_image(img_path)
        return torch.tensor(img)

    def _generate_y(self, img_path):
        img = self._load_image(img_path, front=False)
        return torch.tensor(img)

    def _load_image(self, image_path, front=True):
        if front:
            img_dir = 'D:\\birds_eye_data_final\\data\\front\\'
        else:
            img_dir = 'D:\\birds_eye_data_final\\data\\top\\'

        img = np.load(self.base_path + img_dir + image_path)
        img = img[::2, ::2]
        if not front:
            img = img[3:(img.shape[0] // 2)]
            img = np.rot90(img)
            img = img[2:-2]
            img = img_to_categorical(img, self.needed_classes)

            return img

        img = img[2:-2, 3:-3]
        img = img_to_categorical(img, self.needed_classes)

        return img
