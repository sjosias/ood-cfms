from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch
from glob import glob

class DataFromFile(Dataset):
    '''
        Dataset that generates one of black, white or noisy images
    '''
    def __init__(self, path: str, transform):
        self.files = glob(path + "*")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Convert to tensors
        # get item returns a single item. multiple parallel calls are made.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = read_image(self.files[idx]).type(torch.float32)/255

        # print(image)
        if self.transform:
            image = self.transform(image)

        # print(image)
        return image
