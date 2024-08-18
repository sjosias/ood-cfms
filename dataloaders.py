from torch.utils.data import Dataset, DataLoader
from enum import Enum
import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles, make_blobs
import json

TDataType = Enum('TDataType','annulus half_moon blobs random spiral broken_annulus')

class SanityDataset(Dataset):
    '''
        Dataset that generates one of black, white or noisy images
    '''
    def __init__(self, dataset: str, num_samples: int = 2048, image_dim: int = 1*28*28):
        self.dataset = []
        self.labels = []
        
        if dataset == 'black':

            self.dataset = np.zeros((num_samples, image_dim))
        elif dataset =='white':
            self.dataset = np.ones((num_samples, image_dim))
        elif dataset == 'noise':
            self.dataset = np.random.rand(num_samples, image_dim)
        elif dataset == 'grey':
            self.dataset = np.ones((num_samples, image_dim))/2
        else:
            raise Exception('Please specify the type of sanity dataset init. The options are black, white, noise.')


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert to tensors
        # get item returns a single item. multiple parallel calls are made.
        return torch.from_numpy(self.dataset[idx].astype(np.float32)), []


class ToyDataset(Dataset):
    def __init__(self, dataset_key: TDataType = TDataType.half_moon, num_samples: int = 1000, noise: float = 0.1):
        self.dataset = []
        self.labels = []
        self.noise = noise
        if dataset_key == TDataType.random:
            self.dataset = np.random.rand(num_samples, 2)
            self.labels = np.random.binomial(n = 1, p = 0.5, size = num_samples)

        elif dataset_key == TDataType.annulus:
                self.dataset, self.labels = make_circles(n_samples = num_samples, factor = 0.2, noise = self.noise)
        elif dataset_key == TDataType.half_moon:
                self.dataset, self.labels = make_moons(n_samples = num_samples, noise = self.noise)
        elif dataset_key == TDataType.blobs:
                self.dataset, self.labels = make_blobs(n_samples = num_samples, random_state = 8, centers=2, cluster_std = self.noise)
        # elif dataset_key == TDataType.spiral:
        #         self.dataset, self.labels = self.create_spiral(n_points = num_samples, spiral_degree=2, spiral_gap=4)
        # elif dataset_key == TDataType.broken_annulus:
        #         self.dataset, self.labels = self.create_broken_annulus(n_samples=num_samples, factor = 0.2, noise = self.noise)
        else:
            raise Exception('Please specify the type of toy dataset in ToyDataset init. The options are random, blobs, half_moon, spiral, annulus and broken annulus.')

        min_x, min_y = np.amin(self.dataset, axis=0)
        max_x, max_y = np.amax(self.dataset, axis=0)
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        self.initial_guesses = [[min_x + 0.5*x_range, min_y + 0.5*y_range]]
        intervals = [1/4, 3/4]
        for interval_x in intervals:
            for interval_y in intervals:
                guess = [min_x + interval_x*x_range, min_y + interval_y*y_range]
                self.initial_guesses.append(guess)
        
        self.initial_guesses = np.array(self.initial_guesses)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to tensors
        # get item returns a single item. multiple parallel calls are made.
        label = np.array(self.labels[idx]).astype(np.int32)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        return torch.from_numpy(self.dataset[idx].astype(np.float32)), label





class FeatureVectorsDataset(Dataset):
    '''
        Dataset that gets loaded from precomputed features
    '''
    def __init__(self, file_path: str, dataset_name: str):
        self.dataset = torch.load(file_path)
        self.dataset_name = dataset_name
        
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Convert to tensors
        # get item returns a single item. multiple parallel calls are made.
        # fit method expects data and label
        # data contains 1793 element vector, with 1792 being data, nad the last being the label
        data = self.dataset[idx][:-1]
        label = self.dataset[idx][-1]
        
        return data, label 
    





class LatentDataset(Dataset):
    '''
        Dataset that gets loaded from precomputed features
    '''
    def __init__(self, file_path: str, dataset_name: str, split: str, model: str, dataset_minmaxes: dict):
        

        dataset = "{}{}/{}/{}.pt".format(file_path, dataset_name,model,split)
        print("loading this dataset", dataset, model)
        self.split = split
        self.dataset = torch.load(dataset)
        self.dataset_name = dataset_name
        
        self.data_min = dataset_minmaxes['min']
        self.data_max = dataset_minmaxes['max']

        print("data_min", self.data_min)
        print("data_max", self.data_max)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # data contains a (dataset_size, 4, 28,28) with no label
        data = self.dataset[idx].view(-1)
        # apply transform
        # convert data to (0,1)
        data = data - self.data_min
        data = data / (self.data_max - self.data_min)


        # convert data to (-1,1)
        # print("minmax inside data", torch.min(data), torch.max(data))
        data = (data - 0.5)*2
        return data, []



