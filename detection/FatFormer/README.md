# FatFormer

## Model Zoo

Checkpoint can be found here([baidu](https://pan.baidu.com/s/19wijsvrX0-9Q3dzho8LjUQ?pwd=b759) & [onedrive](https://1drv.ms/u/s!Aqkrc9gPuk8jqZ5Z01EeDlzQIBGFrw?e=F6tdyX) & [Google Drive](https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj/view?usp=sharing)).

### Data Folder Formulation
We expect the directory structure to be the following:
```
path/to/dataset/
├── train/
|   ├── car/ # sub-category of ProGAN
|   |   ├── 0_real # real images
|   |   └── 1_fake # fake images
|   └── ...
└── test/
    ├── AttGAN/ # testing generators
    |   ├── 0_real # real images
    |   └── 1_fake # fake images
    └── ...   
```

## Installation
### Requirements
The code is developed and validated with ```python=3.7.10,pytorch=1.7.1,cuda=11.0```. Higher versions might be as well.

1. Create your own Python environment with [Anaconda](https://www.anaconda.com/download).
```shell
conda env create -f environment.yml
conda activate fatformer
```

2. and other Python packages.
```shell
pip install hashlib pkg_resources tqdm gzip
```

3. To support frequency analysis, you also need to install the `pytorch_wavelets` package following the [pytorch wavelets](https://github.com/fbcotter/pytorch_wavelets) instruction.

```shell
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```

4. Clone this repo.
```shell
git clone https://github.com/Michel-liu/FatFormer.git
cd FatFormer
```

## Inference Guidance

We provide the command to evaluate FatFormer on a single node with 4 gpus.

### Evaluating on the GANs dataset

- `CLIP:ViT-L/14` as backbone, you need first to download the [checkpoint](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and save it into `pretrained` folder under the FatFormer code base.

- for the ckpt, you __also__ need to download the ckpt here [Google Drive](https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj/view?usp=sharing) and put also in the `pretrained`. 

- The final structure tree would be like:

```
FatFormer/
|-- data_example/
|   |-- DM_testdata/
|       |-- test/
|           |-- dalle/
|               |-- 0_real/
|               |-- 1_fake/
|-- models/
|   |-- __pycache__/
|   |-- clip/
|   |-- clip_models.py
|   |-- __init__.py
|-- pretrained/
|   |-- fatformer_4class_ckpt.pth
|   |-- ViT-L-14.pt
|-- pytorch_wavelets/ # third party
|-- utils/
|   |-- __pycache__/
|   |-- dataset.py
|   |-- misc.py
|-- .gitignore
|-- LICENSE
|-- README.md
|-- enviroment.yml
|-- main.py
|-- run.sh
```

The launching script is like:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 29579 \
    main.py \
    --dataset_path <path/to/dataset/> e.g. data_example/DM_testdata \
    --test_selected_subsets 'dalle' '<others>' \
    --eval \
    --pretrained_model <path/to/ckpt/fatformer_4class_ckpt.pth> \
    --num_vit_adapter 3 \
    --num_context_embedding 8
```
- or simple run the `bash run.sh`

### Evaluating on the DMs dataset
You only need to change the `--test_selected_subsets` flag with DMs that you want to evaluate.

## Citation
```bibtex
@inproceedings{liu2024forgeryaware,
  title       = {Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection},
  author      = {Liu, Huan and Tan, Zichang and Tan, Chuangchuang and Wei, Yunchao and Wang, Jingdong and Zhao, Yao},
  booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year        = {2024},
}
```
