import glob
from torchvision import datasets, transforms, models
from torchvision.io import read_image
import torch
from datasets_classifier import DataFromFile
import matplotlib.pyplot as plt
import numpy as np
from classifiers import RGBNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse

BATCH_SIZE = 32



activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook



def obtain_features(net, loader):
    feature_list = []
    # image_list  = []
    for batch_idx, data in enumerate(loader, start=0):
      data = data.to(device)

      model_output = net(data)
      
      feature_list.append(activation['second-fc'].detach().cpu())
      

    return torch.cat(feature_list)


def pairwise_distances(feature_set):
    '''
        Returns pairwise distances between vectors in feature_set.
        if features set is of shape N, D, this returns an nxn matrix
    '''
    
    return torch.cdist(feature_set, feature_set, p =2)**2



def test_distance(feature, feature_set, rhs_distances):
    '''

    args:
        feature: single feature
        feature: set
    '''

    
    # function needs a feature, feature_set, and pairwise distance matrix
    # can compute a list of lhs distances by vectorising
    lhs_distances = torch.linalg.norm(feature - feature_set, dim = 1)**2        

    
    return torch.any(lhs_distances <= rhs_distances).item()
    
    



def precision(sample_features, test_set_features, pairwise_distances):

    #calculate precision for standard
    sample_precision_fk = 0.0
    
    rhs_distances = torch.kthvalue(pairwise_distances, k =2 , dim = 1).values
    
    # m,n = sample_features.shape
    # sample_features = sample_features.T.reshape(1,n,m)
    # m,n = test_set_features.shape
    # test_set_features = test_set_features.reshape(m, n,1)

    # # construct lhs in a vectorised fashion
    # lhs_distances_broadcast = torch.norm(sample_features - test_set_features, dim = 1)**2


    for idx, feature in enumerate(sample_features):
        # print("sample", idx)
        sample_precision_fk += test_distance(feature, test_set_features, rhs_distances)
        # if idx % 1000 == 0:
        #     print("precision for sample", idx)





    sample_precision = sample_precision_fk/sample_features.shape[0]
    print("precision", sample_precision, sample_precision_fk, sample_features.shape[0])




def recall(sample_features, test_set_features, pairwise_distances):

    #calculate precision for standard
    sample_recall_fk = 0.0

    rhs_distances = torch.kthvalue(pairwise_distances, k =2 , dim = 1).values

    
    for idx, feature in enumerate(test_set_features):
            sample_recall_fk += test_distance(feature, sample_features, rhs_distances)
            # if idx % 1000 == 0:
            #     print("recall for sample", idx)

            


    sample_recall = sample_recall_fk/test_set_features.shape[0]
    print("recall", sample_recall, sample_recall_fk, test_set_features.shape)



parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--data', choices= ["cifar10", "svhn"], required=True)
parser.add_argument('--trunc', type=int, default=50000)


args = parser.parse_args()


if __name__ == "__main__":
  
    data = args.data

    print("DATA IS", data)#, test_images.shape)

    test_set_features = torch.load("classifier/saved_models/{}/{}-trainfeatures.pt".format(data,data))
    
    
    
    
    standard_features = torch.load("classifier/saved_models/{}/{}-standardfeatures.pt".format(data,data))
    gmm_features = torch.load("classifier/saved_models/{}/{}-gmmfeatures.pt".format(data,data))
    
    print("standard features shape", standard_features.shape)
    print("gmm features shape", gmm_features.shape)

    assert test_set_features.shape[-1] == 4096
    assert standard_features.shape[-1] == 4096
    assert gmm_features.shape[-1] ==  4096

    assert test_set_features.shape[0] >= 49000
    assert standard_features.shape[0] >= 49000
    assert gmm_features.shape[0] >= 49000

    trunc = args.trunc
    standard_trunc = standard_features[:trunc]
    gmm_trunc = gmm_features[:trunc]
    test_set_trunc = test_set_features[:trunc]
    pair_d = pairwise_distances(test_set_trunc)
    print(pair_d.shape, test_set_trunc.shape)
    print("standard")
    precision(standard_trunc, test_set_trunc, pair_d)
    print("Mixture")
    precision(gmm_trunc, test_set_trunc, pair_d)

    print()
    print("standard")
    pair_d_recall = pairwise_distances(standard_trunc)
    recall(standard_trunc, test_set_trunc, pair_d_recall)

    print("mixture")
    pair_d_recall = pairwise_distances(gmm_trunc)
    recall(gmm_trunc, test_set_trunc, pair_d_recall)