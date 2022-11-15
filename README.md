# LNL-IS
This is the code for our submisssion "Learning with Noisy Labels over Imbalanced Subpopulations"

![Setting](LNLSP.jpg "Setting")

# To obtain the results on Waterbirds or CelebA

You need first to download the public dataset Waterbirds in [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) and CelebA [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

After putting it in, e.g., "../data/", you can generate the noise label using generate_noise.py by directly running

    python3 generate_noise.py

Then you can obtain the same results of our method in our paper by running

    python3 train_waterbirds.py \
        --pretrain \           # using pretrained model as initilization
        --labelconf lv         # how to obtain label confident, 'ERM' label confidence always 1, 'ce' cross-entropy-based label confidence etsimation, and lv ours
        --warm_up 5            # nums of warm up epochs
        --r **                 # noise ratio
        --subpopulation 95     # subpopulation rate 
        --knn **               # k-nearest-neighbor
        --unshifted_val        # validation set has the same imbalance ratio
        --top **               # $\tau$ 
        --root_dir **          # dataset path
or

    python3 train_celebA.py \
        --labelconf lv \       # how to obtain label confident, 'ERM' label confidence always 1, 'ce' cross-entropy-based label confidence etsimation, and lv ours
        --warm_up 5   \        # nums of warm up epochs
        --r **  \              # noise ratio
        --knn **               # k-nearest-neighbor
        --top ** \             # $\tau$
        --root_dir **          # dataset path

You can also obtain the results of ERM baseline or the improved DivideMix by changing the --labelconf lv into 'ERM' or 'ce', respectively.


# Requirement

    python==3.6.8
    scikit-learn==0.23.2
    torch==1.7.0+cu101
    scipy==1.6.2
    Pillow==8.2.0
    pandas==1.2.4
    numpy==1.22.4
