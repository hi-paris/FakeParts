<div align="center">

# AKiRa: Augmentation Kit on Rays for Optical Video Generation

**[Xi Wang](https://triocrossing.github.io/)**, **[Robin Courant](https://robincourant.github.io/info/)**, **[Marc Christie](http://people.irisa.fr/Marc.Christie/)**, **[Vicky Kalogeiton](https://vicky.kalogeiton.info/)**  

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**🌐 Project Webpage**](https://www.lix.polytechnique.fr/vista/projects/2024_akira_wang/)

</div>

---

![Teaser](./assets/teaser.png)

## Overview

As a **TL;DR**:  
We introduce **AKiRa (Augmentation Kit on Rays)** — a ray-space augmentation framework using **Plücker coordinates** that enables video generation models to directly control camera and lens parameters, including:  
**focal length**, **lens distortion**, **aperture**, and **focus point** (for bokeh effects).

---

## 🔬 Benchmark: FlowSim

Evaluating camera-to-video models can be challenging, especially when traditional pose estimation metrics like Absolute Pose Error (APE) and Relative Pose Error (RPE) are unreliable due to short baselines or shaky camera motions.

**FlowSim** offers a robust alternative by computing optical flow similarity between videos, providing a scalable metric to assess camera motion consistency in generated videos.

**Key Features of FlowSim:**
- **Pose-Free Evaluation**: Eliminates the need for explicit camera pose estimation.
- **Robustness**: Effective in scenarios with small translations or unstable camera movements.
- **Scalability**: Suitable for large-scale evaluations of synthetic-to-real generalization.

👉 **Explore the FlowSim repository:**  
**[Triocrossing/FlowSim](https://github.com/Triocrossing/FlowSim)**

## 🚀 Training

```bash
bash ./dist_run.sh configs/train_akira/svd_320_576.yaml N_GPU train_akira.py
```

**📦 Pretrained Checkpoints**:  
🤗[akira checkpoint](https://huggingface.co/xi-wang/akira) on Huggingface

**🙏 Acknowledgment**  
Part of the codebase is adapted from [CameraCtrl](https://github.com/hehao13/CameraCtrl) — many thanks to the authors for their excellent work and their project!

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
