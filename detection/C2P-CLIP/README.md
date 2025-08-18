# C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection

[Chuangchuang Tan](https://scholar.google.com/citations?user=ufR1PmMAAAAJ&hl=zh-CN), [Renshuai Tao](https://rstao-bjtu.github.io/), [Huan Liu](), [Guanghua Gu](), [Baoyuan Wu](), [Yao Zhao](https://scholar.google.com/citations?hl=zh-CN&user=474TbQYAAAAJ), [Yunchao Wei](https://weiyc.github.io/)

Beijing Jiaotong University, YanShan University, CUHK

Reference github repository for the paper [C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection](https://arxiv.org/abs/2408.09647).
```
@article{tan2024c2p,
  title={C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection},
  author={Tan, Chuangchuang and Tao, Renshuai and Liu, Huan and Gu, Guanghua and Wu, Baoyuan and Zhao, Yao and Wei, Yunchao},
  journal={arXiv preprint arXiv:2408.09647},
  year={2024}
}
```

## Environment setup
**Classification environment:** 
We recommend installing the required packages by running the command:
```sh
conda env create -f environment.yml
conda activate c2p
```


## **Detection Inference** 
```
run.sh
```

The dataset structure would be the similar to: FatFormer one:
But instead of inputting the dalle, inputting the dataroot
```
|-- dataroot/
|      |-- dalle/
|               |-- 0_real/
|               |-- 1_fake/
```

Result will compute the acc and ap automatically:
```
==================
      19Test
==================
2025_07_28_01_53_57
(0 dalle       ) acc: 98.55; ap: 99.91
(1 Mean      ) acc: 98.55; ap: 99.91
```