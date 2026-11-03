## Face Forgery Video Detection via Temporal Forgery Cue Unraveling (CVPR 2025)

[Introduction](#introduction) |
[Preparation](#Preparation) |
[Get Started](#get-started) |
[Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.html) |

### Introduction

Face Forgery Video Detection (FFVD) is a critical yet challenging task in determining whether a digital facial video is authentic or forged. Existing FFVD methods typically focus on isolated spatial or coarsely fused spatiotemporal information, failing to leverage temporal forgery cues thus resulting in unsatisfactory performance. We strive to unravel these cues across three progressive levels: momentary anomaly, gradual inconsistency, and cumulative distortion. Accordingly, we design a consecutive correlate module to capture momentary anomaly cues by correlating interactions among consecutive frames. Then, we devise a future guide module to unravel inconsistency cues by iteratively aggregating historical anomaly cues and gradually propagating them into future frames. Finally, we introduce a historical review module that unravels distortion cues via momentum accumulation from future to historical frames. These three modules form our Temporal Forgery Cue Unraveling (TFCU) framework, sequentially highlighting spatial discriminative features by unraveling temporal forgery cues bidirectionally between historical and future frames. Extensive experiments and ablation studies demonstrate the effectiveness of our TFCU method, achieving state-of-the-art performance across diverse unseen datasets and manipulation methods.

### Preparation

#### 1. Environment and Dependencies:

This project is implemented with Python version >= 3.10 and CUDA version >= 11.3.

It is recommended to follow the steps below to configure the environment:
```
conda create -n tfcu python=3.10
conda activate tfcu
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

#### 2.Data Preparation:

Before training, follow the steps below to prepare the data:
1. Download datasets.
   
2. Frame and Landmarks Extraction: Extract frames and landmarks from the video files.
   
3. Face Alignment and Cropping: Referring to the [FTCN](https://github.com/yinglinzheng/FTCN), RetinaFace was chosen for facial recognition, followed by cropping and alignment procedures. When multiple faces appear in the video, tracking the face with the longest appearance time for preservation.

### Quickly Inference

Download weights from [Baidu Cloud(code: ffvd)](https://pan.baidu.com/s/1CvnxPqZ9I8KrEvE9IQRJ4A) and put it into 'checkpoints/Final_TFCU_Model/ckpt'.

Infer a single video: Run the ```python Inference_demo.py```.

###  Evaluation

Download weights from [Baidu Cloud(code: ffvd)](https://pan.baidu.com/s/1CvnxPqZ9I8KrEvE9IQRJ4A) and put it into 'checkpoints/Final_TFCU_Model/ckpt' . Then run:

```
bash test.sh 0 1 12345 checkpoints/Final_TFCU_Model/video_level_c_lm.yaml
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Celeb-DF</th>
<th valign="bottom">DFDC</th>
<th valign="bottom">FFIW</th>
<th valign="bottom">Checkpoints</th>
<tr><td align="left">Ours</td>
<td align="center">93.18%</td>
<td align="center">86.05%</td>
<td align="center">91.27%</td>
<td align="center"><a href="https://pan.baidu.com/s/1CvnxPqZ9I8KrEvE9IQRJ4A">Baidu(code: ffvd)</a></td>
</tbody></table>


### Citation

```
@InProceedings{Guo_2025_CVPR,
    author    = {Guo, Zonghui and Liu, Yingjie and Zhang, Jie and Zheng, Haiyong and Shan, Shiguang},
    title     = {Face Forgery Video Detection via Temporal Forgery Cue Unraveling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {7396-7405}
}
```