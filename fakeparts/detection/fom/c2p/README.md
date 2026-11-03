# C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection [![arXiv](https://img.shields.io/badge/arXiv-2408.09647-b31b1b.svg)](https://arxiv.org/abs/2408.09647)


[Chuangchuang Tan](https://scholar.google.com/citations?user=ufR1PmMAAAAJ&hl=zh-CN), [Renshuai Tao](https://rstao-bjtu.github.io/), [Huan Liu](), [Guanghua Gu](), [Baoyuan Wu](), [Yao Zhao](https://scholar.google.com/citations?hl=zh-CN&user=474TbQYAAAAJ), [Yunchao Wei](https://weiyc.github.io/)

Beijing Jiaotong University, YanShan University, CUHK


:star: If our code is helpful to you, please help star this repo. Thanks! :hugs:

## News 🆕
- [Pretrained models & Text links & Dataset link](https://drive.google.com/drive/folders/1WZStlW2zpH85NZit1-JADMvzovEEZEj9?usp=sharing)

# **Overall Pipeline**
<p align="center">
<img src="./assets/C2P-CLIP.png" width="950px" alt="overall pipeline", align="center">
</p>

---

## 🛠️ Installation
### 1) Main Environment (Training & Detection)

```bash
conda create -n c2pclip python=3.10.14 -y
conda activate c2pclip
pip install -r requirements.txt
```

## 📂 Data Preparation

- Prepare your dataset (e.g., **GenImage**, **UniversalFakeDetect**).
- Download **Genimage_CNNDetection_CLIP_prefix_caption.tar.gz** from the provided [Google Drive link](https://drive.google.com/drive/folders/1WZStlW2zpH85NZit1-JADMvzovEEZEj9?usp=sharing).
- Download CLIP weights (ViT-L/14) from [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14).

---

## 🚀 Usage

### 1) Training

Train C2P-CLIP on GenImage and UniversalFakeDetect.


```bash
conda activate c2pclip

./train_genimage.sh

./train_UniversalFakeDetect.sh
```

### 2) Inference / Testing

```bash
conda activate c2pclip

python inference.py \
  --dataroot ./datasets/GenImage/test/ \
  --model_path ./checkpoints/c2p_clip_genimage/last_model.pth
```


### 3) Feature Analysis (Decoding & Visualization)

```bash
conda activate c2pclip

# Decode features to text
python decode_clipfeature_image.py \
  --image_path ./assets/DALLE/DALLE_2_Cowboy_In_Swamp_Close_Up_Outpaint_1.png \
  --cal_detection_feat

# Visualization (t-SNE)
CUDA_VISIBLE_DEVICES=1 python draw_tsne_kmean.py \
  --draw_data_path ./tsne_png \
  --image_path ./stylegan_tsne_data  \
  --save_name stylegan_test \
  --legend stylegan-bedroom-real stylegan-bedroom-fake stylegan-car-real stylegan-car-fake stylegan-cat-real stylegan-cat-fake \
  --do_extract --do_fit --draw_text 0
```



## 📝 Citation

If you find this code or paper helpful, please cite:

```bibtex
@inproceedings{tan2025c2p,
  title={C2p-clip: Injecting category common prompt in clip to enhance generalization in deepfake detection},
  author={Tan, Chuangchuang and Tao, Renshuai and Liu, Huan and Gu, Guanghua and Wu, Baoyuan and Zhao, Yao and Wei, Yunchao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7184--7192},
  year={2025}
}
```

## 🙏 Acknowledgments

This repository borrows partially from the [CLIPCap](https://github.com/rmokady/CLIP_prefix_caption), [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).
