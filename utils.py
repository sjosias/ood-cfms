import sklearn.datasets
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from dataloaders import SanityDataset, FeatureVectorsDataset, LatentDataset
import torch.nn.functional as nnf
from torch.utils.data import random_split
import json

def get_batch(batch_size):

    data, labels = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)

    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])

    data = torch.tensor(data).type(torch.float32).to(device)
    return data, labels



def reshape(x: torch.Tensor) -> torch.Tensor:
        # x.view(-1) to make shape ([batch_size, dim])
    # reshape data so that hutchinson's trace approximator will work
    return x.view(-1)

# def preprocess(x):
#     # Follows:
#     # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
#     # obtained from: https://github.com/y0ast/Glow-PyTorch/blob/master/datasets.py
#     n_bits = 8

    
#     x = x * 255  # undo ToTensor scaling to [0,1]
#     n_bins = 2 ** n_bits
#     # if n_bits < 8:  
#     #     x = torch.floor(x / 2 ** (8 - n_bits))
#     x = x / n_bins - 0.5

#     return x

def preprocess(x, train = True):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
    # obtained from: https://github.com/y0ast/Glow-PyTorch/blob/master/datasets.py
    
    x = (x * 255. + torch.rand_like(x)) / 256.
    
    x = x #- 0.5

    return x



# def preprocess_encoder(x, train = True):
#     # Follows:
#     # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
#     # obtained from: https://github.com/y0ast/Glow-PyTorch/blob/master/datasets.py
    
#     x = (x * 255. + torch.rand_like(x)) / 256.
    
#     x = x #- 0.5

#     return x


def post_process(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
    # obtained from: https://github.com/y0ast/Glow-PyTorch/blob/master/datasets.py
    n_bits = 8

    
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    # if n_bits < 8:  
    #     x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x

class ShufflePatches(object):
  def __init__(self, patch_size):
    self.patch_size = patch_size

  def __call__(self, x):
    # divide the batch of images into non-overlapping patches
    # SJ we seem to lose batch dimension when loading images

    x = x.reshape(1,*x.shape)
    

    while True:
        u = nnf.unfold(x, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        # permute the patches of each image in the batch
        
        pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
        
        # fold the permuted patches back together
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        
        if not torch.equal(x, f):
            break
    
    return f[0]


RGB_TRANSFORMS_TEST = transforms.Compose(
            [transforms.ToTensor(),
            preprocess,
            transforms.Lambda(reshape)])

RGB_TRANSFORMS_TRAIN = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Lambda(reshape)])

GRAY_TRANSFORMS_TEST =  transforms.Compose([transforms.ToTensor(), 
                                        preprocess,
                                        transforms.Lambda(reshape)
                                        ])


TRANSFORMS_SHUFFLE = lambda patch_size: transforms.Compose([transforms.ToTensor(), 
                                        preprocess,
                                        ShufflePatches(patch_size),
                                        transforms.Lambda(reshape)
                                        ])

GRAY_TRANSFORMS_TRAIN = transforms.Compose([transforms.ToTensor(),                                         
                                        transforms.Lambda(reshape)
                                        ])


FEATURES_RGB = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

FEATURES_GREY = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])



RGB_TO_GREY = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            preprocess,
            transforms.Lambda(reshape)])


DATA_DIM = {
    'fashionmnist': 1*28*28,
    'mnist': 1*28*28,
    "cifar10": 3*32*32,
    "svhn": 3*32*32
}




def get_loader(batch_size: int = 32, 
               data: str = "mnist", 
               training_rate: float = 1.0, 
               test_shuffle: bool = False, 
               train = True, 
               shuffle_patch_size = -1,):
    """
        return data loader (parameterise for different datasets)

    """ 
    batch_size = batch_size
    valset = None
    val_loader = None
    # if train:
    #     transform_gray = GRAY_TRANSFORMS_TRAIN
    #     transform_rgb = RGB_TRANSFORMS_TRAIN
    # else:
    # always use preprocess for now
    if shuffle_patch_size == -1:
        transform_gray = GRAY_TRANSFORMS_TEST
        transform_rgb = RGB_TRANSFORMS_TEST

    else:
        print("SHUFFLING Pixels")
        # room for big error
        transform_gray = TRANSFORMS_SHUFFLE(patch_size=shuffle_patch_size) # divides into 8 patches
        transform_rgb = TRANSFORMS_SHUFFLE(patch_size=shuffle_patch_size) # divides into 8 patches
    
    

    

    if data == "mnist":
        trainset = datasets.MNIST("../data", train=True, download=True, transform=transform_gray)
        testset = datasets.MNIST("../data", train=False, download=True, transform=transform_gray)    
        trainset, valset = random_split(
            trainset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    elif data == "fashionmnist":
        trainset = datasets.FashionMNIST("../data",train=True,download=True,transform=transform_gray)
        testset = datasets.FashionMNIST("../data",train=False,download=True,transform=transform_gray)
        trainset, valset = random_split(
            trainset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
 
    elif data == "cifar10":
        print("loading cifar10 and train is ", train)
        trainset = datasets.CIFAR10("../data",train=True,download=True,transform=transform_rgb)            
        testset = datasets.CIFAR10("../data",train=False,download=True,transform=transform_rgb)
        trainset, valset = random_split(
                trainset, [45000, 5000], generator=torch.Generator().manual_seed(42) 
        )

    elif data == "svhn":
        trainset = datasets.SVHN("../data",split='train',download=True,transform=transform_rgb)            
        testset = datasets.SVHN("../data",split="test",download=True,transform=transform_rgb)
        trainset, valset = random_split(
            trainset, [60_000, 13_257], generator=torch.Generator().manual_seed(42)
        )

    elif data =='black' or data =='white' or data =='noise' or data == 'grey':
        trainset = SanityDataset(dataset = data, num_samples = 2048, image_dim  = DATA_DIM["fashionmnist"])
        testset = SanityDataset(dataset = data, num_samples = 2048, image_dim  = DATA_DIM["fashionmnist"])
    

    if training_rate < 1.0:
        # added a type check because some datasets differ in implementation (ints vs tensor)
        # this works on cifar10 and fashionmnist atm
        if data == "svhn":
            targ = np.array([label if type(label) == int else label.item() for label in trainset.labels]) 
        else:
            targ = np.array([label if type(label) == int else label.item() for label in trainset.targets]) 
        target_idx = np.arange(len(targ))
        y1_idx, _ = train_test_split(target_idx, test_size=1-training_rate, random_state=42, stratify=targ)
        trainset = torch.utils.data.Subset(trainset, indices=y1_idx)


    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # loader
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=test_shuffle, drop_last=True
    )

    if valset:
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    return train_loader, test_loader, val_loader



def normalise(x):
    # print("data shape, ", x.shape)
    mean = torch.mean(x, dim = 0)
    std = torch.std(x, dim=0)

    # print("mean and var shape, ", mean.shape, std.shape)


    return (x - mean)/std



def toy_data(num_samples: int = 512, data: str =  'moons'):
    
    
    data, labels = sklearn.datasets.make_moons(n_samples=num_samples, noise=0.1)
        
    data = data.astype("float32")
    data = data * 2 + np.array([-1, -0.2])

    normalised_data = normalise(torch.tensor(data))

    return normalised_data, labels


