
<div align="center">

<h1>
Framer 
</h1>

<div align="center"> <img src='assets/logo/framer.png' style="height:150px"></img></div>

<h3><a href="https://arxiv.org/abs/2410.18978">Framer: Interactive Frame Interpolation</a></h3>



[Wen Wang](https://github.com/encounter1997)<sup>1,2</sup>, &nbsp; [Qiuyu Wang](https://scholar.google.com/citations?user=VRsy9v8AAAAJ)<sup>2</sup>, &nbsp; [Kecheng Zheng](https://zkcys001.github.io)<sup>2</sup>, &nbsp; [Hao Ouyang](https://ken-ouyang.github.io/)<sup>2</sup>, &nbsp; [Zhekai Chen](https://github.com/Aziily)<sup>1</sup>, &nbsp; [Biao Gong](https://scholar.google.com/citations?user=BwdpTiQAAAAJ)<sup>2</sup>, &nbsp; [Hao Chen](https://scholar.google.com/citations?user=FaOqRpcAAAAJ)<sup>1</sup>, <br>[Yujun Shen](https://shenyujun.github.io)<sup>2</sup>, &nbsp;  [Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University, China](https://www.zju.edu.cn/english/), &nbsp; <sup>2</sup>[Ant Group](https://www.antgroup.com/en)

<br>


<a href='https://arxiv.org/abs/2410.18978'><img src='https://img.shields.io/badge/arXiv-2410.18978-b31b1b.svg'></a> &nbsp;
<a href='https://aim-uofa.github.io/Framer'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=4MPGKgn7jRc'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/spaces/wwen1997/Framer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Demo-blue'></a><br>

<br>

</div>



## 💡 Changelog
- [2024/12/20]. Thanks to [kijai](https://github.com/kijai) for the ComfyUI Support at [ComfyUI-FramerWrapper](https://github.com/kijai/ComfyUI-FramerWrapper)!
- [2024/12/19]. Release the checkpoints.
- [2024/10/28]. Huggingface Gradio Demo is now available [here](https://huggingface.co/spaces/wwen1997/Framer)!
- [2024/10/25]. Launch the project page and upload the arXiv preprint.


  
## 🔆 TL;DR

We propose Framer, a more controllable and interactive frame interpolation method that allows users to produce smoothly transitioning frames between two images by customizing the trajectory of selected keypoints, enhancing control and handling challenging cases. 

### Main Claims
The proposed method, Framer, provides interactive frame interpolation, allowing users to customize transitions by tailoring the trajectories of selected keypoints.
This approach mitigates the ambiguity of image transformation, enabling much finer control of local motions and improving the model's ability to handle challenging cases (e.g., objects with differing shapes and styles).
Framer also includes an "autopilot" mode that automatically estimates keypoints and refines trajectories, simplifying the process and enabling motion-natural and temporally coherent results.

### Methodology
This work utilizes a large-scale pre-trained image-to-video diffusion model (Stable Video Diffusion) as the base model. It introduces additional end-frame conditioning to facilitate video interpolation and incorporates a point trajectory controlling branch for user interaction. 

### Key Results
Framer outperforms existing frame interpolation methods in terms of visual quality and natural motion, particularly in cases involving complex motions and significant appearance changes.
Quantitative evaluation using FVD (Fréchet Video Distance) demonstrates superior performance compared to other methods.
User studies show a strong preference for Framer's output, highlighting its effectiveness in producing realistic and visually appealing results.


## Showcases

Note the videos are spatially compressed. We refer readers to the [project page](https://aim-uofa.github.io/Framer) for the original videos.

### 1. Video Interpolation with User-Interaction

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Input Trajectory & Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_0.png width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/real_dog_00/temp_1_20240925-165335_mode2.gif width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_1.png width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_0.png width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/real_dog_01/temp_1_20240925-165555_mode2.gif width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_1.png width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_0.png width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/real_dog_02/temp_1_20240925-170223_mode2.gif width="250">
  </td>
  <td>
    <img src=assets/001_results_drag/input_frames/dog_1.png width="250">
  </td>
  </tr>

</table>


### 2. Image Morphing with User-Interaction

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Input Trajectory & Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/002_results_morphing/input_frames/dragon1.png width="250">
  </td>
  <td>
    <img src=assets/002_results_morphing/dragon12/temp_1_20240921-112645_mode2.gif width="250">
  </td>
  <td>
    <img src=assets/002_results_morphing/input_frames/dragon2.png width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/002_results_morphing/input_frames/dragon2.png width="250">
  </td>
  <td>
    <img src=assets/002_results_morphing/dragon23/temp_1_20240921-111112_mode2.gif width="250">
  </td>
  <td>
    <img src=assets/002_results_morphing/input_frames/dragon3.png width="250">
  </td>
  </tr>

</table>


</table>


### 3. Video Interpolation without User-Input Control

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/003_direct_interp/input_images/mokey_0.png width="250">
  </td>
  <td>
    <img src=assets/003_direct_interp/monkey_eating_interp_16-43_wocontrol.gif width="250">
  </td>
  <td>
    <img src=assets/003_direct_interp/input_images/mokey_1.png width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/003_direct_interp/input_images/movie_0.jpg width="250">
  </td>
  <td>
    <img src=assets/003_direct_interp/laugh.gif width="250">
  </td>
  <td>
    <img src=assets/003_direct_interp/input_images/movie_1.jpg width="250">
  </td>
  </tr>

</table>


### 4. Novel View Synthesis

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/004_results_nvs/input_images/000001.jpg width="250">
  </td>
  <td>
    <img src=assets/004_results_nvs/scan.gif width="250">
  </td>
  <td>
    <img src=assets/004_results_nvs/input_images/000002.jpg width="250">
  </td>
  </tr>

</table>



### 5. Cartoon and Sketch Interpolation

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/005_results_toon/input_images/74880_401.mp4_00-00.png width="250">
  </td>
  <td>
    <img src=assets/005_results_toon/tooncarft_car.gif width="250">
  </td>
  <td>
    <img src=assets/005_results_toon/input_images/74880_401.mp4_00-01.png width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/005_results_toon/input_images/sketcg1_frame0001.png width="250">
  </td>
  <td>
    <img src=assets/005_results_toon/tooncraft_sketcg1_frame0001.png_to_tooncraft_sketch1_frame0016.png.gif width="250">
  </td>
  <td>
    <img src=assets/005_results_toon/input_images/sketch1_frame0016.png width="250">
  </td>
  </tr>

</table>



### 6. Time-lapsing Video Generation.

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Start Image</td>
        <td>Interpolation Results</td>
        <td>End Image</td>
    </tr>

  <tr>
  <td>
    <img src=assets/006_results_chron/inputs/fafa1.png width="250">
  </td>
  <td>
    <img src=assets/006_results_chron/fafa12.gif width="250">
  </td>
  <td>
    <img src=assets/006_results_chron/inputs/fafa2.png width="250">
  </td>
  </tr>

</table>


## Local Inference

### Installation

```bash
conda create -n framer python=3.8 -y
conda activate framer
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install -r requirements.txt
```


### Prepare weights

Download pretrained [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and [Framer](https://huggingface.co/wwen1997/framer_512x320) weights and put them in `./checkpoints`, as follows.

```bash
checkpoints/
|─ stable-video-diffusion-img2vid-xt
└─ framer_512x320
```

### Run gradio locally

```bash
python app.py
```

Some example inputs are presented in `./examples`, enjoy the interactive frame interpolation!



## 📖 Citation BibTeX
Please consider citing our paper if our code is useful:
```bib
@article{wang2024framer,
  title={Framer: Interactive Video Interpolation},
  author={Wang, Wen and Wang, Qiuyu and Zheng, Kecheng and Ouyang, Hao and Chen, Zhekai and Gong, Biao and Chen, Hao and Shen, Yujun and Shen, Chunhua},
  journal={arXiv preprint https://arxiv.org/abs/2410.18978},
  year={2024}
}
```

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](LICENSE). For commercial use, please contact [C Shen](mailto:chhshen@gmail.com).


## 😉 Acknowledgements

* Our implementation is based on [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend) and [DragAnything](https://github.com/showlab/DragAnything).
* Thanks to [kijai](https://github.com/kijai) for the [ComfyUI Support](https://github.com/kijai/ComfyUI-FramerWrapper).
