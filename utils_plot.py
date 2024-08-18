import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "GyrePagella",
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.handlelength": 1.2
})
# plt.rcParams['legend.handlelength']=0.2


# import tikzplotlib
black = "#000000"
orange = "#E69F00"
sky_blue = "#56B4E9"
bluish_green = "#009E73"
yellow = "#F0E442"
blue = "#0072B2"
vermillion = "#D55E00"
reddish_purple = "#CC79A7"



darker_lightblue = "#22a7f0"
lighter_lightblue = "#a7d5ed"
darker_lightred = "#de6e56"
lighter_lightred = "#e1a692"


opacity_in = (0.5,)
opacity_ood = (1.5*opacity_in[0],)
torgb = lambda h: tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))
orange_rgba = torgb(orange[1:]) + opacity_ood
bluish_green_rgba = torgb(bluish_green[1:]) + opacity_in
sky_blue_rgba = torgb(sky_blue[1:]) + opacity_in
blue_rgba = torgb(blue[1:]) + opacity_in
vermillion_rgba = torgb(vermillion[1:]) + opacity_ood
black_rgba = torgb(black[1:]) + opacity_in
reddish_purple_rgba = torgb(reddish_purple[1:]) + opacity_in
darker_lightblue_rgba = torgb(darker_lightblue[1:]) + opacity_in
##########################################################3
### Load FAshionMNIST
###
##########################################################
LIKELIHOODS_DATA = {
    'fashionmnist': "mnist",
    "mnist": "fashionmnist",
    "cifar10": "svhn",
    "svhn": "cifar10",
    "grey-cifar10": "grey-svhn",
    "grey-svhn": "grey-cifar10"
}




def create_dict(data, base, shuffle_patch_size = -1, model_mode = None, repeat = 0):

    in_data = data
    if shuffle_patch_size == -1:
        ood_data = LIKELIHOODS_DATA[in_data]
    else:
        ood_data = "{}-shuffle-{}".format(in_data, shuffle_patch_size)


    # model_mode could be one of 
    # 1. efficientnet
    # 2. pretrained_encoder
    # 3. raw 
    if model_mode:  
        prefix = "saved_models/base/{}/{}/likelihoods/{}/{}/".format(base, in_data, model_mode ,repeat)
        # print("loading this models likelihoods", prefix)   
    elif shuffle_patch_size > 0:
        prefix = "saved_models/base/{}/{}/likelihoods/".format(base, in_data)
    else:
        # fix this for later
        prefix = "saved_models/base/{}/{}/likelihoods/".format(base, in_data)

    # print("loading models with preix", prefix, "in", in_data, "ood", ood_data)
    in_train_nll = [] #np.loadtxt(prefix + in_data + "_train.npy")
    in_test_nll = np.loadtxt(prefix + in_data + "_test.npy")
    in_test_base = np.loadtxt(prefix + in_data + "_test_base.npy")
    in_test_diff = np.loadtxt(prefix + in_data + "_test_diff.npy")

    out_test_nll = np.loadtxt(prefix + ood_data + "_test.npy")
    out_base = np.loadtxt(prefix + ood_data + "_test_base.npy")
    out_diff = np.loadtxt(prefix + ood_data + "_test_diff.npy")


    # print(in_data + "train", len(in_train_nll), min(in_train_nll), max(in_train_nll))
    # print(in_data + "test", len(in_test_nll), min(in_test_nll), max(in_test_nll))
    # print(ood_data + "test", len(out_test_nll), min(out_test_nll), max(out_test_nll))
    # print()

    return {
        'in-train-likelihood': in_train_nll,
        'in-test-likelihood': in_test_nll,
        'in-test-base': in_test_base,
        'in-test-diff': in_test_diff,
        'out-test-likelihood': out_test_nll,
        'out-test-base': out_base,
        'out-test-diff': out_diff,
    }






def get_datasets(base, model_mode = None, repeat = 0):
    return [
        create_dict(data='mnist',  base = base, model_mode= model_mode, repeat=repeat),
        create_dict(data='fashionmnist', base = base, model_mode = model_mode, repeat=repeat)
        ], [create_dict(data='cifar10',  base = base, model_mode = model_mode, repeat=repeat),
        create_dict(data="svhn",  base = base, model_mode = model_mode, repeat=repeat)
        ]

def create_sanity_dict(sanity_data, in_data, in_data_dict, base):

    
    # prefix = "saved_models/base/"+base+"/"+in_data+"/likelihoods/"

    prefix = "saved_models/base/{}/{}/likelihoods/{}/{}/".format(base, in_data, "raw-sanity" ,0)

    out_test_nll = np.loadtxt(prefix + sanity_data + "_test.npy")
    out_base = np.loadtxt(prefix + sanity_data + "_test_base.npy")
    out_diff = np.loadtxt(prefix + sanity_data + "_test_diff.npy")
    
    print(sanity_data + "sanity", len(out_test_nll), min(out_test_nll), max(out_test_nll))
    print()

    return  {
        'in-train-likelihood': in_data_dict['in-train-likelihood'],
        'in-test-likelihood': in_data_dict['in-test-likelihood'],
        'in-test-base': in_data_dict['in-test-base'],
        'in-test-diff': in_data_dict['in-test-diff'],
        'out-test-likelihood': out_test_nll,  # black likelkhood
        'out-test-base': out_base,
        'out-test-diff': out_diff,
    }





DATA_DIM = {
    'fashionmnist': 1*28*28,
    'mnist': 1*28*28,
    "cifar10": 3*32*32,
    "svhn": 3*32*32
}

DATA_DIM_FEAT = 1792


def sanity_check_plot(data, ax, sanity, out_test_lik, bin_width_log, DENSITY, gen_bins, LW, idx = 0, fc= None, shuffle_patch_size = -1):
    if shuffle_patch_size == -1:
        label_ = LIKELIHOODS_DATA[data]
    else:
        label_ = "{} patches".format((32//shuffle_patch_size)**2)

    if len(sanity) == 0:
        print(ax)
        if idx == -1:
            ax.hist(out_test_lik,  bins=gen_bins(out_test_lik, bin_width_log), density=DENSITY, label = label_, lw=LW, fc=fc, edgecolor='black')
        else:
            ax[idx].hist(out_test_lik,  bins=gen_bins(out_test_lik, bin_width_log), density=DENSITY, label = label_, lw=LW, fc=fc, edgecolor='black')

    else:
        if np.std(out_test_lik) == 0:
            ax.axvline(x=np.mean(out_test_lik), ls='--', label = sanity + " test",c ='black')

        else:
            ax.hist(out_test_lik,  bins=gen_bins(out_test_lik, bin_width_log), density=DENSITY, label = sanity + " test", lw=LW, fc=fc, edgecolor='black')


def plot_constant(data, data_dict, bin_widths, xlims, plot_dim = (1.9,1) , base = "standard"):
    ood_col = orange_rgba #if base == 'standard' else vermillion_rgba
    fig, ax = plt.subplots(1,1, figsize=plot_dim)

    black = data_dict["black"]['out-test-likelihood']
    white = data_dict["white"]['out-test-likelihood']
    grey = data_dict["grey"]['out-test-likelihood']
    noise = data_dict["noise"]['out-test-likelihood']
    in_test_lik = data_dict["in_dist"]["in-test-likelihood"]
    # print("whjite", white)
    gen_bins = lambda nparray, width: np.arange(min(nparray), max(nparray) + width, width)

    # print(noise)
    LW = 0.75
    ax.hist(in_test_lik,  bins=gen_bins(in_test_lik, bin_widths[0]), density=False, label = "in-dist", lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    ax.hist(noise,  bins=gen_bins(noise, bin_widths[1]), density=False, label = "noise", lw=LW, fc=ood_col, edgecolor='black')
    ax.axvline(x=np.mean(black), ls='--', label = "black",c ='black')
    ax.axvline(x=np.mean(white), ls='--', label = "white", c =reddish_purple)
    # ax.axvline(x=np.mean(grey), ls='--', label = "grey", c ='gray')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.65),
            ncol=2, fancybox=False, shadow=False)
    ax.set_xlim(xlims)
    ax.get_yaxis().set_ticks([])

    # ax.set_xlabel(r"log p(x)")
    plt.savefig("plots/{}_{}_sanity.pdf".format(data, base), bbox_inches='tight', dpi=300)


    


def compute_bpd(dataset, base, num_repeats = 3):
    bpd_list_in = []
    bpd_list_out = []
    bpd_f = lambda logp: -(np.mean(logp)/DATA_DIM[dataset] - np.log(256))/(np.log(2)) # second/working
    for i in range(num_repeats):
        loglik_dict = create_dict(data=dataset,  base = base, model_mode= 'raw', repeat=i)
        in_test_lik = loglik_dict['in-test-likelihood']
        out_test_lik = loglik_dict['out-test-likelihood']
        bpd_list_in.append(bpd_f(in_test_lik))
        bpd_list_out.append(bpd_f(out_test_lik))
    # print("in", bpd_list_in)
    # print("out", bpd_list_out)

    print("in-dataset {}: {:0.4f} {:0.4f}".format(dataset, np.mean(bpd_list_in), np.std(bpd_list_in)))
    print("out-dataset {}: {:0.4f} {:0.4f}".format(LIKELIHOODS_DATA[dataset], np.mean(bpd_list_out), np.std(bpd_list_out)))


def plot_loglik(data, data_dict, bin_width_log, bin_width_change, xlim, sanity = "", plot_dim = (1.87, 1.2), xlims = None, base= "standard", use_features = False):
    '''
     Plot decomposed likelihoods
    '''
    # fig, ax = plt.subplots(3,1, figsize=(2,6))

    # plt.tight_layout(h_pad=7)


  
    DENSITY = False
    # bpd = lambda logp: -((np.sum(logp) / (len(logp)*DATA_DIM[data])) - np.log(256))/np.log(2) first
    # glow?
    if use_features:
        bpd = lambda logp: -(np.mean(logp)/DATA_DIM_FEAT)/(np.log(2))    
    else:
        bpd = lambda logp: -(np.mean(logp)/DATA_DIM[data] - np.log(256))/(np.log(2)) # second/working
    # bpd = lambda logp: -(np.mean(logp)/(DATA_DIM[data]))/np.log(2) # own  after subtracting log 256 in likelihood

    #  logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    # bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    # ffjord
    
    # bpd = lambda logp: -np.mean(logp)/DATA_DIM[data]/np.log(2) # second/working
    # logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches


    # bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)

    
    ood_col = orange_rgba #if base == 'standard' else vermillion_rgba

    gen_bins = lambda nparray, width: np.arange(min(nparray), max(nparray) + width, width)
    
    in_train_lik  = data_dict['in-train-likelihood']
    in_test_lik   = data_dict['in-test-likelihood']
    in_test_base  = data_dict['in-test-base']
    in_test_diff  = data_dict['in-test-diff']

    out_test_lik  = data_dict['out-test-likelihood']
    out_test_base = data_dict['out-test-base']
    out_test_diff = data_dict['out-test-diff']

   

    bits_per_dim_in_train = bpd(in_train_lik)
    bits_per_dim_in_test = bpd(in_test_lik)
    bits_per_dim_out_test = bpd(out_test_lik)
    LW = 0.75

    print("bits per dim "+data+" train", bits_per_dim_in_train)
    print("bits per dim "+data+" test", bits_per_dim_in_test)
    if len(sanity) == 0:
        print("bits per dim "+LIKELIHOODS_DATA[data]+" test", bits_per_dim_out_test)
    else:
        print("bits per dim "+sanity+" test", bits_per_dim_out_test)
    print()
    fig, ax = plt.subplots(1,1, figsize=plot_dim)

    # ax.hist(in_train_lik,  bins=gen_bins(in_train_lik, bin_width_log[0]), density=DENSITY, label = data + "tr"  , lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    ax.hist(in_test_lik,  bins=gen_bins(in_test_lik, bin_width_log[0]), density=DENSITY, label = data, lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    sanity_check_plot(data, ax, sanity, out_test_lik, bin_width_log[0], DENSITY, gen_bins, LW, idx = -1 , fc=ood_col)
    # sanity_check_plot(data, ax, sanity, out_test_lik, bin_width, DENSITY, gen_bins, LW, idx = -1 , fc=ood_col, shuffle_patch_size = shuffle_patch_size)

    if xlims is not None:
        ax.set_xlim(xlims[0])
    ax.set_xlabel(r"$\log p(x)$")
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.45),
            ncol=2, fancybox=False, shadow=False)
    ax.get_yaxis().set_ticks([])

    plt.savefig("plots/{}_{}_loglik.pdf".format(data, base), bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1,1, figsize=plot_dim)

    # # # base likelihood
    ax.hist(in_test_base,  bins=gen_bins(in_test_base, bin_width_log[1]), density=DENSITY, label = data, lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    # ax[1].hist(out_test_base,  bins=gen_bins(out_test_base, bin_width_log), density=DENSITY, label = LIKELIHOODS_DATA[data] + " test", lw=LW, fc=orange_rgba, edgecolor='black')
    sanity_check_plot(data, ax, sanity, out_test_base, bin_width_log[1], DENSITY, gen_bins, LW, idx = -1 , fc=ood_col)
    
    if xlims is not None:
        ax.set_xlim(xlims[1])
    # ax.set_xlabel(r"$\log p(z)$")
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.45),
            ncol=2, fancybox=False, shadow=False)
    # plt.rcParams['legend.handlelength']=0.2

    ax.get_yaxis().set_ticks([])
    plt.savefig("plots/{}_{}_baselik.pdf".format(data, base), bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1,1, figsize=plot_dim)

    # # # diff likelihood - volume time
    ax.hist(-1*in_test_diff,  bins=gen_bins(-1*in_test_diff, bin_width_log[2]), density=DENSITY, label = data, lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    # ax[2].hist(out_test_diff,  bins=gen_bins(out_test_diff, bin_width_change), density=DENSITY, label = LIKELIHOODS_DATA[data] + " test", lw=LW, fc=orange_rgba, edgecolor='black')
    sanity_check_plot(data, ax, sanity, -1*out_test_diff, bin_width_log[2], DENSITY, gen_bins, LW, idx = -1 , fc=ood_col)
    ax.set_xlabel(r"negative change logp")
    if xlims is not None:
        ax.set_xlim(xlims[2])
    # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.45),
            ncol=2, fancybox=False, shadow=False)
    ax.get_yaxis().set_ticks([])

    plt.savefig("plots/{}_{}_loglikchange.pdf".format(data, base), bbox_inches='tight', dpi=300)

    # if len(sanity) == 0:
    #     plt.savefig("plots/"+ data + "plots.pdf", bbox_inches="tight")

def unpack(arr):
    return arr[0], arr[1]


def compute_signed_wasserstein(in_dist, ood_dist):
    # determine longest one
    # do a random subset if one is longer
    # compute means to determine sign
    # compute distance
    in_length, ood_length = len(in_dist), len(ood_dist)
    
    num_in_wasserstein = np.amin((in_length, ood_length))
    idx = np.random.permutation(num_in_wasserstein)

    u_values, v_values = in_dist[idx], ood_dist[idx]
    assert len(u_values) == len(v_values)
    sign = 1 if np.mean(u_values)  > np.mean(v_values)   else -1
    
    return sign*wasserstein_distance(u_values, v_values)


def signed_bhattacharya_distance(in_dist, ood_dist):
    in_mean, in_std = np.mean(in_dist), np.std(in_dist)
    out_mean, out_std = np.mean(ood_dist), np.std(ood_dist)

    first_term =(in_mean - out_mean)**2 / (in_std**2 + out_std**2)
    second_term = (in_std**2 + out_std**2)/(2*in_std*out_std)

    sign = 1 if in_mean > out_mean else -1

    return sign*(0.25*first_term + 0.5*second_term)
    



def plot_datasets_loglik(datasets, data_dicts, bin_widths, figsize, xlims = None, sanity="", base = 'standard', shuffle_patch_size = -1, model_mode="", save_fig = True):
    '''
        plot likelihoods for datasets
    '''



  
    DENSITY = False
  
    # bpd = lambda logp: -(np.mean(logp)/DATA_DIM[data] - np.log(256))/(np.log(2)) # second/working
    # bpd = lambda logp: -(np.mean(logp)/DATA_DIM[data] - np.log(256))/(np.log(2)) # second/working
  
    LW = 0.75
    ood_col = orange_rgba if base == 'standard' else vermillion_rgba

    gen_bins = lambda nparray, width: np.arange(min(nparray), max(nparray) + width, width)
    # fig, ax = plt.subplots(1,2, figsize=figsize)

    

    for idx, (data, data_dict, bin_width) in enumerate(zip(datasets, data_dicts,bin_widths)):
        fig, ax = plt.subplots(1, figsize=figsize)
        # in_train_lik  = data_dict['in-train-likelihood']
        in_test_lik   = data_dict['in-test-likelihood']

        out_test_lik  = data_dict['out-test-likelihood']
        

        min_len = min(len(in_test_lik), len(out_test_lik))
        idx_shuffle = np.random.permutation(min_len)
        # in_train_lik = in_train_lik[idx_shuffle]
        in_test_lik = in_test_lik[idx_shuffle] 
        out_test_lik = out_test_lik[idx_shuffle] 

        # if use_features:
        was_dist_test = signed_bhattacharya_distance(in_test_lik, out_test_lik)
        # was_dist_train = compute_signed_wasserstein(in_train_lik, in_test_lik)
        # print("EMD for training data: {} is : {:.2f}".format(data, was_dist_train))
        print("Bhatt for test data: {} is : {:.2f}".format(data, was_dist_test))
        
        # print("data", data, len(out_test_lik))
        
        # ax.hist(in_train_lik,  bins=gen_bins(in_train_lik, bin_width), density=DENSITY, label =  data + " tr"  , lw=LW, fc=reddish_purple_rgba, edgecolor='black')
        ax.hist(in_test_lik,  bins=gen_bins(in_test_lik, bin_width), density=DENSITY, label =   data, lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
        sanity_check_plot(data, ax, sanity, out_test_lik, bin_width, DENSITY, gen_bins, LW, idx = -1 , fc=ood_col, shuffle_patch_size = shuffle_patch_size)
        if xlims is not None:
            print("setting", xlims[idx])

            ax.set_xlim(xlims[idx])
        ax.set_xlabel(r"$\log p(x)$")
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticks([])
        if data == 'fashionmnist' and save_fig:
            ax.set_ylim([0, 2091.6])
        elif data == 'cifar10' and save_fig:
            ax.set_ylim([0, 2222.6])
        # ax[index].legend()
        # ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.65),
        #   ncol=1, fancybox=False, shadow=False)

        # suffix = "" if shuffle_patch_size == -1 else "_shuffle-{}".format(shuffle_patch_size)
        # ax.legend()
        suffix = 'raw' if model_mode == None else model_mode
        # print("plots/singledata_{}_{}_{}.pdf".format(base, datasets[idx],suffix))
        if save_fig:
            plt.savefig("plots/singledata_{}_{}_{}.pdf".format(base, datasets[idx],suffix), dpi=300, bbox_inches='tight')
    # plt.savefig("plots/all_data_{}.eps".format(base), dpi=300, bbox_inches='tight')

    # tikzplotlib.save("plots/all_data_{}.tex".format(base))
    plt.show()
    # if len(sanity) == 0:
    #     plt.savefig("plots/"+ data + "plots.pdf", bbox_inches="tight")


def plot_shuffle(ax, data_dicts, gen_bins, bin_width, datasets,LW,shuffle_patch_size,DENSITY, base, ood_cols, sanity):
    in_test_lik   = data_dicts[0]['in-test-likelihood']
    # in test likelihood is the same for both data dicts 0, and 1
    ax.hist(in_test_lik,  bins=gen_bins(in_test_lik, bin_width), density=DENSITY, label = datasets[0], lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
    for idx, (data, data_dict, ps) in enumerate(zip(datasets, data_dicts, shuffle_patch_size)):
        

        out_test_lik  = data_dict['out-test-likelihood']

        
        # print("data", data, len(out_test_lik))
        # ax.hist(in_train_lik,  bins=gen_bins(in_train_lik, bin_width), density=DENSITY, label =  data + " tr"  , lw=LW, fc=darker_lightblue_rgba, edgecolor='black')
        
        sanity_check_plot(data, ax, sanity, out_test_lik, bin_width, DENSITY, gen_bins, LW, idx = -1 , fc=ood_cols[idx], shuffle_patch_size = shuffle_patch_size[idx])
        
        # ax.set_xlabel(r"$\log p(x)$")
        # ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
        ax.get_yaxis().set_ticks([])

        # ax[index].legend()
        if base == "standard":
            ax.legend(loc='upper center',bbox_to_anchor=(1.1, 1.5),
          ncol=3, fancybox=False, shadow=False)
            


def plot_datasets_loglik_shuffle(datasets, data_dicts, bin_width, figsize, xlims = [4000,16000], sanity="", base = 'standard', shuffle_patch_size = -1):
    '''
        plot likelihoods for datasets
        datasets = [cifar10, cifar10]
        data_dicts = [cifar10-16, cifar10-8]
    '''



  
    DENSITY = False
  
    # bpd = lambda logp: -(np.mean(logp)/DATA_DIM[data] - np.log(256))/(np.log(2)) # second/working
    LW = 0.75
    ood_col_1 = orange_rgba # if base == 'standard' else vermillion_rgba
    ood_col_2 = reddish_purple_rgba
    ood_cols = [ood_col_1, ood_col_2]

    gen_bins = lambda nparray, width: np.arange(min(nparray), max(nparray) + width, width)
    # fig, ax = plt.subplots(1,2, figsize=figsize)

    fig, (ax, ax2) = plt.subplots(1,2, figsize=figsize)
    # ax2.get_yaxis().set_ticks([])

    plot_shuffle(ax, data_dicts[0], gen_bins, bin_width, datasets,LW,shuffle_patch_size,DENSITY, base, ood_cols, sanity)
    plot_shuffle(ax2, data_dicts[1], gen_bins, bin_width, datasets,LW,shuffle_patch_size,DENSITY, "gmm", ood_cols, sanity)
    

    plt.savefig("plots/data_standard-gmm_{}_shuffle.pdf".format(base, datasets[-1]), dpi=300, bbox_inches='tight')
    # plt.savefig("plots/data_{}_{}_shuffle.pdf".format(base, datasets[-1]), dpi=300, bbox_inches='tight')
    # plt.savefig("plots/all_data_{}.eps".format(base), dpi=300, bbox_inches='tight')

    # tikzplotlib.save("plots/all_data_{}.tex".format(base))
    plt.show()
    # if len(sanity) == 0:
    #     plt.savefig("plots/"+ data + "plots.pdf", bbox_inches="tight")

