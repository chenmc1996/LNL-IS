# LNL-IS
This is the code for our submisssion "Learning with Noisy Labels over Imbalanced Subpopulations"

![Setting](LNLSP.jpg "Setting")


## To obtain the results on Waterbirds or CelebA

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

## To obtain the results on ANIMAIL-10N

You first need to download the public dataset ANIMAIL-10N in [here](https://dm.kaist.ac.kr/datasets/animal-10n/#:~:text=ANIMAL-10N%20dataset%20contains%205%20pairs%20of%20confusing%20animals,using%20the%20predifined%20labels%20as%20the%20search%20keyword)
Facing a real-world noisy dataset, we don't need to preprocess the label information.
Just run

    python3 train_animal.py \
        --warm_up 15   \       # nums of warm up epochs
        --knn 5000             # k-nearest-neighbor
        --top 0.9 \            # $\tau$
        --root_dir **          # dataset path


## Requirement

    python==3.6.8
    scikit-learn==0.23.2
    torch==1.7.0+cu101
    scipy==1.6.2
    Pillow==8.2.0
    pandas==1.2.4
    numpy==1.22.4
