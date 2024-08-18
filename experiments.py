from models import ResizeUNetModel
import torch
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
# from models import MLPWrapper, MLP, SkipMLP
from training import FlowMatchingTrainer
from torch.utils.data import DataLoader
from utils import get_loader
import wandb
import torch.distributions as D
import argparse
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Required args
parser.add_argument('--data', choices= ["mnist", "fashionmnist", "cifar10", "svhn", 'grey-cifar10', 'grey-svhn'], required=True)
parser.add_argument('--warmup', type=int, default=5000)

parser.add_argument('--base', choices= ["standard", "gmm"], required=True)


# Optional args
parser.add_argument('--train', action='store_true')
parser.add_argument('--shuffle_patch_size', type=int, default=-1)
parser.add_argument('--sanity', action='store_true')
parser.add_argument('--sample', action='store_true')
parser.add_argument('--n_epochs', type=int, default=10) # implement early stopping
parser.add_argument('--batch_size', type=int, default=32) # implement early stopping
parser.add_argument('--test_batch_size', type=int, default=256) # implement early stopping
parser.add_argument('--net_channels', type=int, default=32) # implement early stopping
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--skip', action='store_true')
parser.add_argument('--cov_scale', type=float, default=0.5)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--dim', type=int, default=-1)
parser.add_argument('--reduce_train', action='store_true')


args = parser.parse_args()

# Global variables for data details from args
SINGLE_CHANNEL_28 = ['mnist', 'fashionmnist']
GREY_CHANNEL_32 = ['grey-cifar10', "grey-svhn"]
TRIPLE_CHANNEL_32 = ['cifar10', "svhn"]
LIKELIHOODS_DATA = {
    'fashionmnist': "mnist",
    "mnist": "fashionmnist",
    "cifar10": "svhn",
    "svhn": "cifar10",
    "grey-cifar10": "grey-svhn",
    "grey-svhn": "grey-cifar10"
    
}
DATA_DIM = {
    'fashionmnist': 28,
    "mnist": 28,
    "cifar10": 32,
    "svhn": 32,
}


# sanity check for calculating likelihoods
LIKELIHOODS_DATA_SANITY = ['grey', 'black', "white", 'noise']


SAVE_DIR = "saved_models/base/{}/{}/raw_models/".format(args.base, args.data)
LOAD_DIR = SAVE_DIR
SAVE_DIR_LIKELIHOODS = "saved_models/base/{}/{}/".format(args.base, args.data)
    





print("SAVE DIR", SAVE_DIR)
print("LOAD DIR", LOAD_DIR)

def data_details(data: str = "mnist"):
    """
        returns
        -----------------------------
        data_shape: tuple containing data shape
        data_dim: int describiZng total dimensionality of the data
    """
    
    if data in SINGLE_CHANNEL_28 :
        return (1, 28, 28), 28*28*1
    elif data in TRIPLE_CHANNEL_32:
        return (3, 32, 32), 32*32*3
    elif data in GREY_CHANNEL_32:
        return (1, 32, 32), 32*32*1
    
def warmup_lr(step):
    # lambda function for LR scheduler
    # LR is warmed up to a peak, and then reduced to 1e-8
    return min(step, args.warmup) / args.warmup

def get_base(data_dim, num_components = 10, cov_scale = args.cov_scale):
    if args.base =='standard':
        means = torch.zeros(data_dim, dtype=torch.float32).to(device)
        cov = cov_scale*torch.eye(data_dim, dtype=torch.float32).to(device)

        return D.MultivariateNormal(loc=means,covariance_matrix=cov)
    
    elif args.base == 'gmm':

        # load pre-computed dataset means
        means = torch.load("data_modes/" + args.data + "_modes.pt").to(device)
        means = torch.zeros(means.shape).to(device)
        cov = cov_scale*torch.ones(means.shape).to(device)
        mix = torch.distributions.Categorical(torch.ones(num_components,).to(device))
        comp = D.Independent(D.Normal(means, cov), 1)
        
        return D.mixture_same_family.MixtureSameFamily(mix, comp)

        
def setup():
    """
        Set up trainer
    """

    data_shape, data_dim = data_details(data=args.data)    
    n_channels = data_shape[0]
    LR = args.lr

    diffeq = ResizeUNetModel(dim=data_shape, 
                    image_channel=n_channels,
                    num_channels=args.net_channels,
                    num_res_blocks=1,
                    num_heads=1,
                    num_head_channels=-1,
                    attention_resolutions="16",
                    ).to(device)
      
    # base distribution
    print("setting up prior", data_dim)
    p_z0 = get_base(data_dim)

    sigma = 0
    optimizer = torch.optim.AdamW(diffeq.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)    

    FM = TargetConditionalFlowMatcher(sigma=sigma)

    return FlowMatchingTrainer(model = FM, 
                               vf = diffeq,
                               prior=p_z0, 
                               optimizer = optimizer,
                               scheduler = scheduler, 
                               device=device, 
                               save_dir="saved_models/" +args.data+"/",
                               data_shape= data_shape,
                               data=args.data)



def train_exp(train_loader: DataLoader, val_loader: DataLoader, repeat: int = 1):
    """
        Run experiment to train a model
    """
    trainer = setup()
    
    gen_samples = True # true because we're not using features at all
        
    trainer.fit(loader = train_loader, 
                val_loader = val_loader, 
                n_epochs = args.n_epochs, 
                skip=args.skip, 
                save_dir = SAVE_DIR, 
                test_batch_size = args.test_batch_size, 
                base = args.base, 
                samples = gen_samples)

    prefix = SAVE_DIR + "/{}/".format(repeat)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    
    # TODO Fix model naming convention
    trainer.save_model(save_dir=prefix + "model{}_covariance_{}_{}_epoch.pt".format(repeat, str(args.cov_scale), args.n_epochs), ema_save_dir=prefix + "model{}_covariance_{}_{}_epoch_ema.pt".format(repeat, str(args.cov_scale), args.n_epochs))
    
    
   

def likelihood_filename():
    
    return "raw"
    
def feature_name():
    if args.use_features:
        return "pretrained-classifier"
    elif args.pretrained_encoder:
        return "pretrained-encoder"
    elif args.conv_pixel_encoder:
        return "conv_pixel_encoder"
    elif args.pixel_encoder:
        return "pixel-encoder"
    elif args.basic_pixel_encoder:
        return "basic-pixel-encoder"
    

    
def likelihoods_with_repeat(train_loader, test_loader, repeat):


    p_data = args.data
    print("evaluating likelihoods on in-diustribution data", p_data, "with sfhulle pixels", args.shuffle_patch_size)
    trainer = setup()
    
    prefix = repeat #"" if args.repeats == 1 else repeat



    try:
        # TODO this could cause problems with models trained on raw pixels
        model_path = LOAD_DIR + "{}/model{}_covariance_{}_{}_epoch.pt".format(prefix, repeat,str(args.cov_scale), args.n_epochs)
        print("TRYING TO FIRST LOAD loaded a model from", model_path)
        trainer.load_model(model_path)
    except:
        print("An exception occurred, likely vecuase we are trying to load a model without the epochs in file name, loading this model instead")
        model_path = LOAD_DIR + "{}/model{}_covariance_{}_{}_epoch.pt".format(prefix, repeat,str(args.cov_scale), args.n_epochs)
        trainer.load_model(model_path)
        print("loaded a exception model from", model_path)



    MODEL_MODE = likelihood_filename()
 
    trainer.sample(save_fig=True, save_name = "samples_test_experiment.png")


    if args.shuffle_patch_size > 0:
        trainer.likelihoods_exp(loader=test_loader, 
                                predict_data=p_data, 
                                train = False, 
                                skip = args.skip,  
                                save_dir=SAVE_DIR_LIKELIHOODS, 
                                covariance = args.cov_scale, 
                                shuffle_patch_size = args.shuffle_patch_size, 
                                model_mode= "shuffle-raw", 
                                repeat=repeat)


    elif args.sanity:
        print("SANITY")
        for p_data in LIKELIHOODS_DATA_SANITY:
            _, test_loader, _ = get_loader(batch_size=args.batch_size, 
                                           data=p_data, 
                                           training_rate=1.0, 
                                           train = args.train, 
                                           use_features=args.use_features, 
                                           pretrained_encoder=args.pretrained_encoder)
            # get_loader(batch_size=args.batch_size, data=p_data, training_rate=train_rate, train = args.train, test_shuffle= True, use_features=args.use_features, pretrained_encoder=args.pretrained_encoder)
            trainer.likelihoods_exp(loader=test_loader, 
                                    predict_data=p_data, 
                                    train = False, 
                                    skip = args.skip, 
                                    save_dir=SAVE_DIR_LIKELIHOODS, 
                                    covariance = args.cov_scale, 
                                    shuffle_patch_size = -1, 
                                    model_mode = "raw-sanity", 
                                    repeat=repeat)

    elif args.shuffle_patch_size == -1:
        # use_features to include use_features or pretrained autoencoder
        # should consider doing this for training autoencoder from scratch        
       
        # trainer.likelihoods_exp(loader=train_loader, 
        #                         predict_data= p_data, 
        #                         train = True, 
        #                         skip = args.skip, 
        #                         save_dir=SAVE_DIR_LIKELIHOODS, 
        #                         covariance = args.cov_scale, 
        #                         shuffle_patch_size = -1, 
        #                         model_mode = MODEL_MODE, 
        #                         repeat=repeat)
        
        print("DOING LIKELIHOODS ON TEST DATA")
        trainer.likelihoods_exp(loader=test_loader, 
                                predict_data=p_data, 
                                train = False, 
                                skip = args.skip,  
                                save_dir=SAVE_DIR_LIKELIHOODS, 
                                covariance = args.cov_scale, 
                                shuffle_patch_size = -1, 
                                model_mode= MODEL_MODE, 
                                repeat=repeat)
        
        p_data = LIKELIHOODS_DATA[args.data]

        print("DOING LIKELIHOODS ON OOD DATA", p_data)
        # _, test_loader, _ = get_loader(batch_size=args.batch_size, data=p_data, training_rate=train_rate, train = args.train, test_shuffle= True, use_features=args.use_features, pretrained_encoder=args.pretrained_encoder)
        _, test_loader, _ = get_loader(batch_size=args.batch_size, 
                                       data=p_data, 
                                       training_rate=train_rate, 
                                       train = args.train, 
                                       test_shuffle= True, 
                                       shuffle_patch_size=args.shuffle_patch_size)
        trainer.likelihoods_exp(loader=test_loader, 
                                predict_data=p_data, 
                                train = False, 
                                skip = args.skip, 
                                save_dir=SAVE_DIR_LIKELIHOODS, 
                                covariance = args.cov_scale, 
                                shuffle_patch_size = -1, 
                                model_mode = MODEL_MODE, 
                                repeat=repeat)
    
    


def sampling_with_repeat(repeat = 1):
    trainer = setup()

    prefix = repeat #"" if args.repeats == 1 else repeat
    print("Loading directory", LOAD_DIR)
    try:
        # TODO this could cause problems with models trained on raw pixels
        trainer.load_model(LOAD_DIR + "{}/model{}_covariance_{}_{}_epoch.pt".format(prefix, repeat,str(args.cov_scale), args.n_epochs))
        print("loading the correct model")
    except:
        print("An exception occurred, likely vecuase we are trying to load a model without the epochs in file name, loading this model instead")
        trainer.load_model(LOAD_DIR + "{}/model{}_covariance_{}_{}_epoch.pt".format(prefix, repeat,str(args.cov_scale), args.n_epochs))
    
    # 2. Set up trainer.sample to save to saved_models/base/data/pretrainedencoder_samples/repeat/samples.pt
    samples_list = []
    # generate 50K samples 
    for i in range(int(50000/args.batch_size)):
        samples_dir = "saved_models/base/{}/{}/gen_samples/{}/".format(args.base, args.data, repeat)
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        trainer.sample(save_fig=True, save_individual=True, num_samples=args.batch_size, save_name=samples_dir, batch_idx=i)

        if args.skip and i == 1:
            break

   


if __name__ == '__main__':
    
    # reduced train rate, specifically to test calculate likelihoods
    if args.reduce_train:
        train_rate = 0.1666667
    else:
        train_rate = 1.0

    if not args.sample:
        project_name = "gmm-flows-test"
        name = "train-" + args.data + "-" + args.base if args.train else "likelihood-" + args.data + "-" + args.base
        run = wandb.init(
            # Set the project where this run will be logged
            project=project_name,
            # name of the run
            name=name,
            group="train" if args.train else "likelihoods",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "data": args.data,
                "base": args.base,
                "warmup": args.warmup,
                "skip": args.skip,
                "sample": args.sample,
                "net channels": args.net_channels,
                "cov-scale": args.cov_scale
            },
        )
        
        train_loader, test_loader, val_loader = get_loader(batch_size=args.batch_size, 
                                                           data=args.data, 
                                                           training_rate=train_rate, 
                                                           train = args.train, 
                                                           test_shuffle= True, 
                                                           shuffle_patch_size=args.shuffle_patch_size)

    if args.train:
        for i in range(args.repeats):
            train_exp(train_loader= train_loader, val_loader = val_loader, repeat = i)
    elif args.sample:
        for i in range(args.repeats):
            sampling_with_repeat(repeat = i)
    else:
        for i in range(args.repeats):
            likelihoods_with_repeat(train_loader, test_loader, repeat = i)
