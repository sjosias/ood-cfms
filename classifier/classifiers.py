
import torch.nn as nn
from torch.nn import functional as F
import torch

class RGBNet(nn.Module):
  def __init__(self, input_shape=(3,32,32), num_classes = 10):
    super(RGBNet, self).__init__()
    
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    
    self.pool = nn.MaxPool2d(2,2)

    n_size = self._get_conv_output(input_shape)
    
    self.fc1 = nn.Linear(n_size, 512)
    self.fc2 = nn.Linear(512, num_classes)

    self.dropout = nn.Dropout(0.25)

  def _get_conv_output(self, shape):
    batch_size = 1
    input_ = torch.autograd.Variable(torch.rand(batch_size, *shape))
    output_feat = self._forward_features(input_)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size

  def _forward_features(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    return x
  
  def fc_features(self, x):
    x = self._forward_features(x)
    x = x.view(x.size(0), -1)
    x = self.dropout(x)
    return self.fc1(x)

  def forward(self, x):
    x = F.relu(self.fc_features(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x