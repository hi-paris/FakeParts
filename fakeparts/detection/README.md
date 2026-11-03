# Detection

## Methods, checkpoints, and pipelines

| Family | Method | Venue | Source repository | License |
| --- | --- | --- | --- | --- |
| OoB | CNNDet | CVPR'20 | [peterwang512/CNNDetection](https://github.com/peterwang512/CNNDetection) | CC-BY-NC-SA-4.0 |
| OoB | HiFi | CVPR'23 | [chelsea234/hifi_ifdl](https://github.com/chelsea234/hifi_ifdl) | MIT |
| OoB | TALL | ICCV'23 | [rainy-xu/TALL4Deepfake](https://github.com/rainy-xu/TALL4Deepfake) | MIT |
| OoB | NPR | CVPR'24 | [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) | -- |
| OoB | AIGV | PRCV'24 | [multimediaFor/AIGVDet](https://github.com/multimediaFor/AIGVDet) | CC-BY-NC-4.0 |
| OoB | IML-ViT | NeurIPS'24 | [SunnyHaze/IML-ViT](https://github.com/SunnyHaze/IML-ViT) | MIT |
| OoB | TFCU | CVPR'25 | [zhenglab/TFCU](https://github.com/zhenglab/TFCU) | -- |
| FoM | De-Fake | ArXiv'23 | [zeyangsha/De-Fake](https://github.com/zeyangsha/De-Fake) | MIT |
| FoM | UnivFD | CVPR'23 | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | MIT |
| FoM | FatFormer | CVPR'24 | [Michel-liu/FatFormer](https://github.com/Michel-liu/FatFormer) | Apache-2.0 |
| FoM | D³ | CVPR'25 | [BigAandSmallq/D3](https://github.com/BigAandSmallq/D3) | -- |
| FoM | C2P | AAAI'25 | [chuangchuangtan/C2P](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) | -- |
| FoM | DFD-FCG | CVPR'25 | [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG) | -- |
| VLM | MM-Det | NeurIPS'24 | [SparkleXFantasy/MM-Det](https://github.com/SparkleXFantasy/MM-Det) | Apache-2.0 |
| VLM | FakeVLM | NeurIPS'25 | [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM) | -- |
| VLM | SIDA | CVPR'25 | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) | Llama 2 Community |
| VLM | AFPrompt | ArXiv'25 | [nctu-eva-lab/AntifakePrompt](https://github.com/nctu-eva-lab/antifakeprompt) | BSD-3-Clause |
| VLM | BusterX | NeurIPS'26 | [l8cv/BusterX](https://github.com/l8cv/BusterX) | BSD-3-Clause |
| DiB | DRCT | ICML'24 | [beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT) | -- |
| DiB | NSG-VD | NeurIPS'25 | [ZSHsh98/NSG-VD](https://github.com/ZSHsh98/NSG-VD) | -- |

### Checkpoints

| Method | Official checkpoints / backbone | Mirror |
| --- | --- | --- |
| CNNDet | [blur/JPEG 0.5](https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1) · [0.1](https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1) | [OrkaZeta/CNNDet](https://huggingface.co/OrkaZeta/CNNDet) |
| HiFi | [Google Drive](https://drive.google.com/drive/folders/1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm) | [OrkaZeta/HiFi_IFDL](https://huggingface.co/OrkaZeta/HiFi_IFDL) |
| TALL | -- | [bitmind/tall](https://huggingface.co/bitmind/tall) |
| NPR | [NPR.pth](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/NPR.pth) | -- |
| AIGV | [Google Drive](https://drive.google.com/drive/folders/18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP) · [Google Drive](https://drive.google.com/file/d/1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM) | [OrkaZeta/AIGVDet](https://huggingface.co/OrkaZeta/AIGVDet) |
| IML-ViT | [Google Drive](https://drive.google.com/drive/folders/1Ztyiy2cKJVmyusYMUlwuyPecBefTJCPT) | -- |
| TFCU | [Baidu (ffvd)](https://pan.baidu.com/s/1CvnxPqZ9I8KrEvE9IQRJ4A) · [RetinaFace](https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet0.25_Final.pth) | [OrkaZeta/TFCU](https://huggingface.co/OrkaZeta/TFCU) |
| De-Fake | [Google Drive](https://drive.google.com/drive/folders/1qI7x5iodaCFq0S61LKw4wWjql7cYou_4) · [Google Drive](https://drive.google.com/drive/folders/1SuenxJP10VwArC6zW0SHMUGObMRqQhBD) | -- |
| UnivFD | [FC weights](https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth) | -- |
| FatFormer | [Google Drive](https://drive.google.com/file/d/1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj) · [OneDrive](https://1drv.ms/u/s!Aqkrc9gPuk8jqZ5Z01EeDlzQIBGFrw?download=1) · [Baidu (b759)](https://pan.baidu.com/share/init?surl=9wijsvrX0-9Q3dzho8LjUQ&pwd=b759) | -- |
| D³ | [Classifier](https://github.com/BigAandSmallq/D3/raw/main/ckpt/classifier.pth) | -- |
| C2P | [C2P-CLIP release](https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip) | -- |
| DFD-FCG | [Google Drive](https://drive.google.com/file/d/1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk) | -- |
| MM-Det | [Google Drive](https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy) · [sparklexfantasy/llava-7b-1.5-rfrd](https://huggingface.co/sparklexfantasy/llava-7b-1.5-rfrd) | [OrkaZeta/MMDet](https://huggingface.co/OrkaZeta/MMDet) |
| FakeVLM | [lingcco/fakeVLM](https://huggingface.co/lingcco/fakeVLM) | -- |
| SIDA | [saberzl/SIDA-7B](https://huggingface.co/saberzl/SIDA-7B) · [saberzl/SIDA-13B](https://huggingface.co/saberzl/SIDA-13B) | -- |
| AFPrompt | [Google Drive](https://drive.google.com/file/d/1EUnVG4OZZPXeOyWaa5P590yCKGH-nunQ) · [Google Drive](https://drive.google.com/file/d/1qnZfCknNHgC-Nhlwbab9Jg3x9sOof3gG) | [OrkaZeta/AntifakePrompt](https://huggingface.co/OrkaZeta/AntifakePrompt) |
| BusterX | [l8cv/BusterX-plusplus](https://huggingface.co/l8cv/BusterX-plusplus) | -- |
| DRCT | [ModelScope](https://www.modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/pretrained.zip) | [OrkaZeta/DRCT](https://huggingface.co/OrkaZeta/DRCT) |
| NSG-VD | -- | [OrkaZeta/NSG-VD](https://huggingface.co/OrkaZeta/NSG-VD) |

## Shared weights

| Backbone | Checkpoint | Used by |
| --- | --- | --- |
| CLIP ViT-L/14 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | CLIP-based detectors |
| OpenAI CLIP ViT-L/14 | [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | D³, DFD-FCG, FatFormer |
| BLIP caption | [model_base_capfilt_large.pth](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth) | De-Fake |
| InstructBLIP | [Vicuna-7B backbone](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) | AFPrompt |
| Vicuna-7B | [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3) | AFPrompt |
| SAM ViT-H | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | SIDA |
| 2D-FAN | [2DFAN4](https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar) | DFD-FCG |
| SFD | [s3fd](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) | DFD-FCG |
| Diffusion score model | [256×256 unconditional](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | NSG-VD |

### Pipelines

| Method | Input | Inference pipeline |
| --- | --- | --- |
| CNNDet | Frame | RGB → optional center crop → ImageNet normalization → ResNet-50 → sigmoid |
| HiFi | Frame | 256×256 RGB → HRNet → NLCDetection → maximum manipulation-mask response |
| TALL | Video | 8 × 4 sampled frames → 2×2 thumbnails → TALL-Swin → mean clip logits → softmax |
| NPR | Frame | 256px resize → center crop → ImageNet normalization → NPR ResNet-50 → sigmoid |
| AIGV | Video | 448px RGB + adjacent-frame RAFT flow → two ResNet branches → mean frame/video score |
| IML-ViT | Frame | 1024px pad/crop → ViT + feature pyramid → manipulation mask → maximum response |
| TFCU | Video | RetinaFace alignment + landmarks → 4-frame clips → TFCU → mean frame probability |
| De-Fake | Frame | BLIP caption → CLIP image/text embeddings → fusion classifier → softmax |
| UnivFD | Frame | CLIP ViT-L/14 → linear head → sigmoid |
| FatFormer | Frame | 256px resize → 224px crop → CLIP + forgery-aware adapter → softmax |
| D³ | Frame | 224px RGB → original/shuffled CLIP features → attention head → sigmoid |
| C2P | Frame | CLIP ViT-L/14 → normalized embedding → linear head → sigmoid |
| DFD-FCG | Video | Face tracking + affine crops → overlapping 10-frame clips → FFG-L/14 → mean probability |
| MM-Det | Video | RGB + VQ-VAE reconstruction + LLaVA features → 10-frame windows → mean probability |
| FakeVLM | Frame | LLaVA image processing → FakeVLM → normalized fake/real answer likelihood |
| SIDA | Frame | CLIP image processing → SIDA classification head → P(synthetic) + P(tampered) |
| AFPrompt | Frame | InstructBLIP image processing → authenticity question → yes/no candidate losses → fake score |
| BusterX | Video | Qwen3.5 video processing → boxed A/B answer → binary fake score |
| DRCT | Frame | Resize/pad/crop → ConvNeXt or CLIP → two-class softmax |
| NSG-VD | Video | Sampled frames → diffusion-score features → NSG features → dMMD against real references |
