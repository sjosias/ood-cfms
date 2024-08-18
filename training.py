from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm # progress bar
import torch
from torch.distributions.distribution import Distribution
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

from models import ODEfunc, CNF
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from utils import get_loader
import wandb
from joblib import load
from yaml import safe_load as conf_load
import json
import os
import copy

from sklearn.metrics import accuracy_score





DATA_LABELS = {
            "cifar10": 0,
            "mnist": 1,
            "fashionmnist": 2,
            "svhn": 3
    }

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

LIKELIHOODS_DATA = {
    'fashionmnist': "mnist",
    "mnist": "fashionmnist",
    "cifar10": "svhn",
    "svhn": "cifar10"
}



class FlowMatchingTrainer():
    """Class used to train Flow mathcing ODENets, ConvODENets and ResNets.

    Parameters
    ----------
    model : torchcfm.conditional_flow_matching.ConditionalFlowMathcer

    optimizer : torch.optim.Optimizer instance

    device : torch.device 

    save_dir: str
    """
    def __init__(self, model: ConditionalFlowMatcher, vf: torch.nn,  prior: Distribution, optimizer: Optimizer, scheduler,
        device, save_dir: str = None, data_shape = (1,28,28), data: str = "fashionmnist"):
        # decode=False, ema_decay=0.9999, conv_encoder=False

        self.flow_matcher = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.vf = vf
        self.prior = prior # prior to be used by cnf for loglikelihood
        # self.save_dir = save_dir
        time_length = 1
        train_T = True
        regularization_fns = None
        solver = "dopri5"# 'dopri5'
        self.data_shape = data_shape 
        self.data = data
        
        # total data dimensionality, gets square rooted in ODEnet (vf)
        self.data_dim = data_shape[0]*data_shape[1]*data_shape[2] if len(data_shape) == 3 else data_shape[0]*data_shape[1] 

        # the ODEfunc that computes the ODEnet and divergence terms
        odefunc = ODEfunc(diffeq=vf, divergence_fn="approximate", residual=False)
        # class the does the integration and calculates likelihoods
        self.cnf = CNF(odefunc=odefunc, T=time_length, train_T=train_T, regularization_fns=regularization_fns, solver=solver, data_shape = self.data_shape).to(device)


    def samples_like(self, base, x1, labels):
        '''
            Should return samples the same shape as x1. For the gmm,
            samples should come from components linked to the label.
        '''
        if base == 'standard':
            return torch.randn_like(x1)
        elif base == 'gmm':
            counts = np.bincount(labels)
            max_sample_per_comp = np.max(counts)
            samples = self.prior.component_distribution.sample(sample_shape=(max_sample_per_comp,))
            samples = samples.transpose(0,1).to(device)

            num_comps = samples.shape[0]
            ret_samples = torch.zeros(x1.shape).to(device)
            for comp in range(num_comps):
                comp_idx_labels = torch.argwhere(labels == comp).flatten()
                num_idx = len(comp_idx_labels)
                ret_samples[comp_idx_labels,  :] = samples[comp, :num_idx, :]
            return ret_samples
            

    def train_step(self, data, labels, base):
        self.optimizer.zero_grad()

        x1 = data.to(self.device) 
        x0 = self.samples_like(base, x1, labels).to(self.device)

        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self.vf(t, xt) 
        
        if len(self.data_shape) == 3:
            vt = vt.reshape(-1, self.data_dim) 

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
    @torch.no_grad()
    def val_step(self,data,labels,base):
        x1 = data.to(self.device) 
        x0 = self.samples_like(base, x1, labels).to(self.device)
        
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self.vf(t, xt) 
        
        if len(self.data_shape) == 3:
            vt = vt.reshape(-1, self.data_dim) 

        loss = torch.mean((vt - ut) ** 2)

        return loss.item()


    def fit(self, loader: DataLoader, val_loader: DataLoader, n_epochs: int = 1, skip=False, save_dir="", test_batch_size = 200, base: str = "standard", samples= True):
        # refer to https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/train_cifar10.py#L89
        
        last_val_loss = -1
        for epoch in range(n_epochs):
            epoch_loss = 0.
            for i, (data, labels) in tqdm(enumerate(loader)):
                epoch_loss += self.train_step(data=data, labels=labels, base=base)
                if skip: 
                    print("breaking now as part of testing")
                    break

            epoch_loss = epoch_loss/len(loader)
            
            if epoch % 3 == 0:
                if samples:
                    self.sample(save_fig=True, save_name=save_dir+"trained"+self.data+str(epoch)+".png")

                # run validation
                val_loss = 0.
                for i, (data, labels) in tqdm(enumerate(val_loader)):
                    val_loss += self.val_step(data=data, labels=labels, base=base)


                val_loss = val_loss / len(val_loader)
                last_val_loss = val_loss
                wandb.log({"val/loss": val_loss})


            wandb.log({"train/loss": epoch_loss, "last_lr": self.scheduler.get_last_lr()[0]})

        if samples:
            self.sample(save_fig=True, save_name=save_dir+"trained"+self.data+".png")
        return last_val_loss


    def compute_classifier_scores(self, dataset, num_samples):
        """
            Only applicable if features are used
        """

        # generate samples from the model
        
        samples = self.gen_samples(num_samples).detach().cpu().numpy()
        print("classifer samples", samples.shape)
        # classifier
        # Generate labels from dataset
        # X_train = np.concatenate([cifar10[0],mnist[0], fashionmnist[0], svhn[0]])
        # y_train = np.concatenate([np.ones(cifar10[1].shape)*0, np.ones(mnist[1].shape)*1, np.ones(fashionmnist[1].shape)*2, np.ones(svhn[1].shape)*3])
        clf_loaded = load("saved_models/lda-discriminator/four-class.joblib")
        true_labels  = np.ones(num_samples)*DATA_LABELS[dataset]
        # train data accuracy
        test_predict = clf_loaded.predict(samples)
        
        accuracy = accuracy_score(true_labels, test_predict)
        wandb.log({"classifier-accuracy-test": accuracy, "classifier-label": DATA_LABELS[dataset]})
        
        

    def likelihoods_exp(self, loader: DataLoader, predict_data: str, train: bool = True, skip = False, save_dir="", covariance = 1.0, shuffle_patch_size = False, model_mode="False", repeat = 1):
        """
                Run experiment to compute likelihoods
            """

        if shuffle_patch_size > 0:
            predict_data = "{}-shuffle-{}".format(predict_data, shuffle_patch_size)
            
        # if covariance == 1.0:
            # prefix =  save_dir+"/likelihoods/efficientnet/" if use_features else save_dir+"/likelihoods/raw/"
        prefix = save_dir + "/likelihoods/{}/".format(model_mode)
        # else:
        #     raise Exception("Covariance other than 1.0 is not supported")


        # create directory
        prefix = prefix + "/{}/".format(repeat)
        
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        prefix = prefix + predict_data
        
        print("likelihoods exp save path", prefix , model_mode)

        likelihoods_list, nfe_list, base_likelihood, likelihoodiff  = self.compute_nll(loader, skip=skip, train = train, predict_data=predict_data)
        likelihoods_list = likelihoods_list.detach().cpu().numpy()
        base_likelihood = base_likelihood.detach().cpu().numpy()
        likelihoodiff = likelihoodiff.detach().cpu().numpy()

        fname_likelihoods = prefix + "_train.npy" if train else prefix + "_test.npy"
        fname_nfe = prefix + "_train_nfe.npy" if train else prefix + "_test_nfe.npy"
        fname_base = prefix + "_train_base.npy" if train else prefix + "_test_base.npy"
        fname_diff = prefix + "_train_diff.npy" if train else prefix + "_test_diff.npy"
        print("FILE NAME",fname_likelihoods)
        np.savetxt(fname_likelihoods, likelihoods_list)
        np.savetxt(fname_nfe, nfe_list)
        np.savetxt(fname_base, base_likelihood)
        np.savetxt(fname_diff, likelihoodiff)
        if train:
            name = predict_data + "-train-likelihoods"
        else:
            name = predict_data + "-test-likelihoods"
        artifact = wandb.Artifact(name=name, type="likelihood")
        artifact.add_file(local_path=fname_likelihoods)
        wandb.log_artifact(artifact)



    def save_model(self, save_dir: str, ema_save_dir: str = None):
        """
            Saves model dict
        """        
        torch.save(self.vf.state_dict(), save_dir)


    def load_model(self, load_dir: str):
        """
            Saves model dict
        """
        self.cnf.odefunc.diffeq.load_state_dict(torch.load(load_dir))
        self.cnf.odefunc.diffeq.eval()
        

    
    def gen_samples(self, num_samples):
        base_samples = self.prior.sample(sample_shape=(num_samples,))
        integration_time = torch.tensor([0, 1], dtype=torch.float32).to(device)

        logp_diff_t1 = torch.zeros(base_samples.shape[0], 1).type(torch.float32).to(device)
                
        self.cnf.eval()
        z_t1, _ = self.cnf(z = base_samples, logpz=logp_diff_t1, integration_times=integration_time)
        logp_diff_t1 = torch.zeros(base_samples.shape[0], 1).type(torch.float32).to(device) 
        self.cnf.train()
        print("NFE in gen samples", self.cnf.num_evals())
    
        return z_t1
    
    def decode_samples(self, samples):
        f = open(self.minmax_path) 
        dataset_minmaxes = json.load(f)
        small, big = dataset_minmaxes[self.dataload]['min'], dataset_minmaxes[self.dataload]['max']
        print("in the deocde for gen samples", small, big, samples.shape )
        if self.autoencoder.connected_layer:
            
            latents = samples.view(-1,self.data_shape[0]*self.data_shape[1]*self.data_shape[2])
        else:
            latents = samples.view(-1,self.data_shape[0],self.data_shape[1],self.data_shape[2])
        print("in the deocde for gen samples", latents.shape )

        latents = latents/2 + 0.5
        shifted_latents = latents*(big-small) + small
        decoded_images = self.autoencoder.decode(shifted_latents)
        return decoded_images
    
    def log_decoded_training(self, train_loader):
        print("logging training images")
        for idx, batch in enumerate(train_loader):
            
            image_input = batch[0].to(device)            
            break
            # if idx == 1: break
        print("shape", image_input.shape)
        decoded_images = self.decode_samples(image_input)
        grid_decoded = make_grid(
            decoded_images[:100].clip(0,1), value_range=(0,1), padding=0, nrow=10, normalize=True
        )
        image_wandb_decoded = wandb.Image(grid_decoded, caption="Decoded training latents")
        wandb.log({"Decoded training latents": image_wandb_decoded})



    def sample(self, save_fig= False, save_individual = False, num_samples = 10, save_name: str = None, batch_idx = 0):
        z_t1 = self.gen_samples(num_samples=num_samples)
        v_range = (0, 1)
        if save_fig:
            
            if save_individual:                
                for k, image in enumerate(z_t1):
                    save_image(image.clip(*v_range), fp=save_name + "{}.png".format(num_samples*batch_idx + k), value_range=v_range, padding=0, normalize=True)

            else:
                if len(z_t1.shape) == 2: 
                    z_t1 = z_t1.view(-1,self.data_shape[0], self.data_shape[1], self.data_shape[2])
                

                grid = make_grid(
                    z_t1.clip(-1,1), value_range=v_range, padding=0, nrow=num_samples, normalize=True
                )
                image_wandb = wandb.Image(grid, caption="samples for model")
                wandb.log({"samples": image_wandb})


        return z_t1

    def log_prob(self, z_t0: torch.tensor, logp_diff_t0: torch.tensor) -> torch.tensor:
        '''
            Function to calculate log probability given  solutions to the reverse ODE
        '''
        
        # prior changed to mixture same family
        # this calculates a weighted mixture over the gmm, or just the standard
        log_prob_base = self.prior.log_prob(z_t0.view(z_t0.shape[0],-1)).to(device)
        # trying to subtract datadim log 256 to account for conversion
        return log_prob_base - logp_diff_t0.view(-1), log_prob_base

    def compute_nll(self, loader: DataLoader, t0: int = 0, t1: int = 1, train: bool = False, skip: bool = False, predict_data: str =""):
        """
            Function to loop through a dataloader and return a histogram of likelihoods
            Plot histograms for MNIST, FASHIONMNIST, CIFAR10, SVHN


            - Should modify networks to accept image dimension / take out hardcode
            - Train on FASHIONMNIST, plot FASHIONMNIST / MNIST likelihoods
            - Train on CIFAR10, plot CIFAR10 / SVHN likelihoods

            Parameters
            ----------------------
            loader: DataLoader
                Loader for data (could be for training and testing)

            base_likelihood: bool
                Variable to indicate whether to return base_likelihood or not

        """
        integration_time = torch.tensor([t1, t0], dtype=torch.float32).to(device)

        log_probs = []
        nfe = []
        log_base = []
        logp_change = []
        total_samples = 0
        # print("Length of ", len(loader))
        self.cnf.eval()
        for batch_idx, (image, _) in tqdm(enumerate(loader)):
            # print("before image to device")

            image = image.to(device)
            print("image shape", image.shape)
            logp_diff_t1 = torch.zeros(image.shape[0], 1).type(torch.float32).to(device)
            
            z_t0, logp_diff_t0 = self.cnf(z = image, logpz=logp_diff_t1, integration_times=integration_time)
        
            logp_x, logp_x_base = self.log_prob(z_t0, logp_diff_t0)

            log_probs.append(logp_x)
            log_base.append(logp_x_base)
            logp_change.append(logp_diff_t0)
            nfe.append(self.cnf.num_evals())
            print("NFE in flow matchign compute_nll", nfe)
            if skip and batch_idx == 2:
                print("skipping likelihoods")
                break

            total_samples += image.shape[0]
            # we want to cut all datasets that have samples over 10000
            # this includes all the training sets, and the svhn train and svhn test set
            # print("predict data is", predict_data, "train is", train)
            if (train or predict_data == "svhn") and total_samples >= 10_000:
                print("stopping likelihood computations on data set with train", predict_data, train)
                break
            
        # Have to compile the base loglikelihood and diffs to list as well
        return torch.cat(log_probs), nfe, torch.cat(log_base), torch.cat(logp_change)
        
        


