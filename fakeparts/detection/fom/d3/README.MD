# ​D³: Scaling Up Deepfake Detection by Learning from Discrepancy

[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.04584)
[![Conference](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com/)

Official implementation of "​D³: Scaling Up Deepfake Detection by Learning from Discrepancy" (CVPR 2025). This work reveals two challenges when scaling up generators in the training set and proposes a dual-branch framework to improve generalization in multi-generator deepfake detection by leveraging discrepancy signals from distorted images.

<div style="text-align:center;">
  <img src="Figs/framework.png" alt="D³ Framework" style="width:90%;">
</div>

## Abstract

<div style="text-align: justify;">
The boom of Generative AI brings opportunities entangled with risks and concerns. Existing literature emphasizes the generalization capability of deepfake detection on unseen generators, significantly promoting the detector's ability to identify more universal artifacts. This work seeks a step toward a universal deepfake detection system with better generalization and robustness. We do so by first scaling up the existing detection task setup from the one-generator to multiple-generators in training, during which we disclose two challenges presented in prior methodological designs and demonstrate the divergence of detectors' performance. Specifically, we reveal that the current methods tailored for training on one specific generator either struggle to learn comprehensive artifacts from multiple generators or sacrifice their fitting ability for seen generators (i.e., *In-Domain* (ID) performance) to exchange the generalization for unseen generators (i.e., *Out-Of-Domain* (OOD) performance). Therefore, detectors' performance gradually diverges during the scaling up of generators. To tackle the above challenges, we propose our ​**Discrepancy Deepfake Detector (D³)** framework, whose core idea is to deconstruct the universal artifacts from multiple generators by introducing a parallel network branch that takes a distorted image feature as an extra discrepancy signal and supplement its original counterpart. Extensive scaled-up experiments demonstrate the effectiveness of ​**D³**, achieving a 5.3% accuracy improvement in the OOD testing compared to the current SOTA methods while maintaining the ID performance.
</div>


## Scaling-up Process

We design an experiment to show the diversed performance of different methods when gradually adding generators into the training pool. 

<div style="text-align:center;">
  <img src="Figs/scale-up-generators.png" alt="scaling-up process" style="width:90%;">
</div>


## Results
| Method       | ID Accuracy | OOD Accuracy | Total Accuracy | Publication |
|--------------|-------------|--------------|----------------|-------------|
| CNNDet       | 93.3%       | 69.9%        | 79.2%          | CVPR20      |
| Patchfor     | 97.9%       | 78.9%        | 86.5%          | ECCV20      |
| LNP          | 88.1%       | 71.9%        | 78.4%          | ECCV22      |
| UFD          | 86.6%       | 81.4%        | 83.5%          | CVPR23      |
| UCF          | 91.7%       | 75.0%        | 81.7%          | ICCV23      |
| NPR          | **98.6%**       | 78.7%        | 86.6%          | CVPR24      |
| ​D³ (Ours)   | ​96.6%​   | ​**86.7%​**    | ​**90.7%​**      | CVPR25      |


## Detection Robustness

Our method demonstrates excellent robustness. Here are experiments on Gaussian blurring ranging from 0 to 2 and JPEG compression ranging from 30 to 100 to verify the robustness of different methods.

<div style="text-align:center;">
  <img src="Figs/Robustness.png" alt="detection robustness" style="width:90%;">
</div>

## Train and Inference

📦 1. Environment Setup

```bash
cd D3 
conda env create -f environment.yml -n D3 
conda activate D3 
```
🚀 2. Training Instructions

Before training, replace the folder path under the `TODO` placeholders in `train.py` with your dataset folder path.

Execution:
```bash
python train.py --name=train_d3 --arch=CLIP:ViT-L/14 --checkpoints_dir=path/to/save/checkpoints --fix_backbone --head_type=attention --batch_size=256 --shuffle --patch_size=14
```
🔮 3. Inference Instructions

Our pretrained weight is in folder ckpt/classifier.pth.

Before inferencing, replace the folder path under the `TODO` placeholders in `validate_for_robustness.py` with your dataset folder path

Execution:
```bash
python validate_for_robustness.py
```

📂 4. Combined Dataset

Our composite dataset merges resources from:
[UFD](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main)
and
[GenImage](https://github.com/GenImage-Dataset/GenImage).

## Citation
If you find our work helpful in your research, please cite it using the following:
```bibtex
@inproceedings{yang2025d3,
  title={D3: Scaling Up Deepfake Detection by Learning from Discrepancy},
  author={Yang, Yongqi and Qian, Zhihao and Zhu, Ye and Russakovsky, Olga and Wu, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```





