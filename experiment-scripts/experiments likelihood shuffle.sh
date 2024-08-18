#!/bin/sh

num_epochs="100"
cov_scale="1.0"

# Standard base patchs 4
# time python experiments.py --data fashionmnist  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 14 
# time python experiments.py --data mnist  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 14 
time python experiments.py --data cifar10  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002  --n_epochs 150 --shuffle_patch_size 16 --skip
# time python experiments.py --data svhn  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 16 

# Standard base patchs 16
# time python experiments.py --data fashionmnist  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 7 
# time python experiments.py --data mnist  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 7  
# time python experiments.py --data cifar10  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002  --n_epochs 100 --shuffle_patch_size 8 
# time python experiments.py --data svhn  --cov_scale $cov_scale --base standard --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 8 



# GMM base pachces 4
# time python experiments.py --data fashionmnist  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 14 
# time python experiments.py --data mnist  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 14 
# time python experiments.py --data cifar10  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002  --n_epochs 100 --shuffle_patch_size 16 
# time python experiments.py --data svhn  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 16 


# GMM base pachces 16
# time python experiments.py --data fashionmnist  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 7 
# time python experiments.py --data mnist  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 7  
# time python experiments.py --data cifar10  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002  --n_epochs 100 --shuffle_patch_size 8 
# time python experiments.py --data svhn  --cov_scale $cov_scale --base gmm --batch_size 512 --net_channels 128 --lr 0.0002 --n_epochs 100 --shuffle_patch_size 8 

