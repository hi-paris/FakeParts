from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from fakeparts.naming import DETECTORS, FAMILIES, version_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = PROJECT_ROOT / "ckpt" / "detection"

DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 4
DEFAULT_DEVICE = "cuda"
DEFAULT_THRESHOLD = 0.5

# Real-video prediction CSVs used by AUC. Detection writes one pool per source
# under ``real_<source>.<detector_version>.csv``; AUC combines these configured
# sources and samples a balanced control set for each fake task-method group.
REAL_REFERENCE_CLASSES = ("tenk_youtube", "youtube_vos_2019")

CSV_FIELDS = {
    "frame": ("sample_id", "video", "rel_path", "frame", "class", "method",
              "label", "score", "prediction", "error"),
    "video": ("sample_id", "video", "rel_path", "class", "method", "label",
              "score", "prediction", "error"),
}


@dataclass(frozen=True)
class Checkpoint:
    """One file required by a detector."""
    path: Path
    url: str | None = None
    hf_repo: str | None = None
    hf_filename: str | None = None
    required_files: tuple[str, ...] = ()
    gdrive_id: str | None = None
    gdrive_filename: str | None = None


@dataclass(frozen=True)
class DetectorConfig:
    """Adapter settings, including its initial batch for each free-VRAM tier.

    VRAM-tier values are starting points measured after loading the model; the
    runner reduces a batch after CUDA OOM and keeps that smaller size.
    """
    family: str
    module: str
    class_name: str
    output_level: str = "frame"
    checkpoints: tuple[Checkpoint, ...] = ()
    batch_size: int = DEFAULT_BATCH_SIZE
    implemented: bool = False
    checkpoint_variants: dict[str, tuple[Checkpoint, ...]] = field(default_factory=dict)
    default_variant: str | None = None
    batch_size_by_free_vram_gb: tuple[tuple[float, int], ...] = ()
    batch_unit: str = "frames"
    adapter_class: str | None = None
    _registry_id: str = field(default="", compare=False, repr=False)

    def version_name(self, version_id: str) -> str:
        return version_name(self._registry_id, version_id)

    @property
    def full_name(self) -> str:
        return DETECTORS[self.class_name_id][0]

    @property
    def short_name(self) -> str:
        return DETECTORS[self.class_name_id][1]

    @property
    def class_name_id(self) -> str:
        return self._registry_id

    @property
    def family_full_name(self) -> str:
        return FAMILIES[self.family][0]

    @property
    def family_short_name(self) -> str:
        return FAMILIES[self.family][1]

    @property
    def ckpt(self) -> Path | None:
        return self.checkpoints[0].path if self.checkpoints else None

    def batch_size_for_free_vram(self, free_vram_gb: float | None) -> int:
        """Choose a method-specific starting batch from currently free VRAM."""
        if not self.batch_size_by_free_vram_gb:
            return self.batch_size
        if free_vram_gb is None:
            return self.batch_size_by_free_vram_gb[0][1]
        selected = self.batch_size_by_free_vram_gb[0][1]
        for minimum_gb, size in self.batch_size_by_free_vram_gb:
            if free_vram_gb < minimum_gb:
                break
            selected = size
        return selected


def _ckpts(*paths: str) -> tuple[Checkpoint, ...]:
    return tuple(Checkpoint(CKPT_ROOT / path) for path in paths)


def _hf_ckpt(
    path: str, repo: str, filename: str, *, url: str | None = None,
    gdrive_id: str | None = None, gdrive_filename: str | None = None,
) -> Checkpoint:
    return Checkpoint(CKPT_ROOT / path, url=url, hf_repo=repo,
                      hf_filename=filename, gdrive_id=gdrive_id,
                      gdrive_filename=gdrive_filename)


def _gdrive_ckpt(path: str, file_id: str, filename: str | None = None) -> Checkpoint:
    return Checkpoint(CKPT_ROOT / path, gdrive_id=file_id, gdrive_filename=filename)


_AFPROMPT_BLIP2_CKPT = Checkpoint(
    CKPT_ROOT / "AntiFakePrompt/blip2/instruct_blip_vicuna7b_trimmed.pth",
    "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/"
    "InstructBLIP/instruct_blip_vicuna7b_trimmed.pth",
)
_AFPROMPT_VICUNA_CKPT = Checkpoint(
    PROJECT_ROOT / ".cache/models/vicuna-7b-v1.3",
    hf_repo="lmsys/vicuna-7b-v1.3",
    required_files=(
        "config.json", "generation_config.json", "pytorch_model.bin.index.json",
        "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
        "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json",
    ),
)


def _nsg_ckpts(head: str) -> tuple[Checkpoint, Checkpoint]:
    return (
        _hf_ckpt("NSG-VD/Checkpoints/256x256_diffusion_uncond.pt",
                 "OrkaZeta/NSG-VD", "Checkpoints/256x256_diffusion_uncond.pt",
                 url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"),
        Checkpoint(PROJECT_ROOT / "fakeparts/detection/dib/nsg_vd/ckpts" / Path(head).name),
    )


DET_MODELS: dict[str, DetectorConfig] = {
    "aigvdet": DetectorConfig(
        "oob", "aigvdet", "AIGVDet", "video",
        (_hf_ckpt("AIGVDet/raft_model/raft-things.pth", "OrkaZeta/AIGVDet",
                  "raft_model/raft-things.pth", gdrive_id="1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM"),
         _hf_ckpt("AIGVDet/checkpoints/original.pth", "OrkaZeta/AIGVDet",
                  "checkpoints/original.pth", gdrive_id="18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP",
                  gdrive_filename="original_aug.pth"),
         _hf_ckpt("AIGVDet/checkpoints/optical.pth", "OrkaZeta/AIGVDet",
                  "checkpoints/optical.pth", gdrive_id="18JO_YxOEqwJYfbVvy308XjoV-N6fE4yP",
                  gdrive_filename="optical_aug.pth")),
        batch_size=8, implemented=True,
        batch_size_by_free_vram_gb=((0, 4), (8, 8), (12, 16), (20, 32), (32, 48), (64, 64))),
    "cnndet": DetectorConfig(
        "oob", "cnndet", "CNNDetection", adapter_class="CNNDet",
        checkpoints=(_hf_ckpt(
            "CNNDetection/weights/blur_jpg_prob0.5.pth", "OrkaZeta/CNNDet",
            "weights/blur_jpg_prob0.5.pth",
            url="https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"),),
        batch_size=128, implemented=True,
        checkpoint_variants={
            "blur_jpg_prob0_1": (_hf_ckpt(
                "CNNDetection/weights/blur_jpg_prob0.1.pth", "OrkaZeta/CNNDet",
                "weights/blur_jpg_prob0.1.pth",
                url="https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1"),),
            "blur_jpg_prob0_5": (_hf_ckpt(
                "CNNDetection/weights/blur_jpg_prob0.5.pth", "OrkaZeta/CNNDet",
                "weights/blur_jpg_prob0.5.pth",
                url="https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"),),
        },
        default_variant="blur_jpg_prob0_5",
        batch_size_by_free_vram_gb=((0, 32), (8, 64), (12, 128), (20, 256), (32, 384), (64, 512)),
    ),
    "hifi": DetectorConfig("oob", "hifi", "HiFi-Net", checkpoints=(
        _hf_ckpt("HiFi-IFDL/weights/HRNet/750001.pth", "OrkaZeta/HiFi_IFDL",
                 "weights/HRNet/750001.pth", gdrive_id="1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm",
                 gdrive_filename="HRNet/750001.pth"),
        _hf_ckpt("HiFi-IFDL/weights/NLCDetection/750001.pth", "OrkaZeta/HiFi_IFDL",
                 "weights/NLCDetection/750001.pth", gdrive_id="1v07aJ2hKmSmboceVwOhPvjebFMJFHyhm",
                 gdrive_filename="NLCDetection/750001.pth"),
    ), batch_size=64, implemented=True, adapter_class="HiFiNet",
        batch_size_by_free_vram_gb=((0, 8), (8, 16), (12, 32), (20, 64), (32, 96), (64, 128))),
    "iml_vit": DetectorConfig(
        "oob", "iml_vit", "IML-ViT", checkpoints=(
            _gdrive_ckpt("IML-ViT/checkpoints/iml-vit_checkpoint.pth",
                         "1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7"),),
        batch_size=1, implemented=True,
        checkpoint_variants={
            "iml_vit_release": (_gdrive_ckpt(
                "IML-ViT/checkpoints/iml-vit_checkpoint.pth",
                "1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7"),),
        }, default_variant="iml_vit_release", adapter_class="IMLViT",
        batch_size_by_free_vram_gb=((0, 1), (8, 2), (12, 4), (20, 8), (32, 12), (64, 16)),
    ),
    "npr": DetectorConfig("oob", "npr", "NPR", checkpoints=_ckpts(
        "NPR/NPR.pth"), batch_size=128, implemented=True,
        checkpoint_variants={"npr_release": _ckpts("NPR/NPR.pth")},
        default_variant="npr_release",
        batch_size_by_free_vram_gb=((0, 32), (8, 64), (12, 128), (20, 256), (32, 384), (64, 512))),
    "tall": DetectorConfig(
        "oob", "tall", "TALL", "video",
        (_hf_ckpt("TALL/tall_trainFF_testCDF.pth", "bitmind/tall",
                  "tall_trainFF_testCDF.pth"),),
        1, implemented=True, batch_unit="videos"),
    "tfcu": DetectorConfig(
        "oob", "tfcu", "TFCU", "video",
        (
            _hf_ckpt("TFCU/checkpoints/Final_TFCU_Model/ckpt/Final.tar",
                     "OrkaZeta/TFCU", "checkpoints/Final_TFCU_Model/ckpt/Final.tar"),
            Checkpoint(
                CKPT_ROOT / "TFCU/retinaface/mobilenet0.25_Final.pth",
                "https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet0.25_Final.pth",
            ),
        ), 16, implemented=True, batch_unit="frames",
        batch_size_by_free_vram_gb=((0, 4), (8, 8), (12, 16), (20, 32), (32, 48), (64, 64)),
    ),
    "c2p": DetectorConfig("fom", "c2p", "C2P", checkpoints=(Checkpoint(
        CKPT_ROOT / "C2P/weights/C2P_CLIP_release_20240901.pth",
        "https://www.now61.com/f/95OefW/C2P_CLIP_release_20240901.zip",
    ),), implemented=True,
        batch_size_by_free_vram_gb=((0, 8), (8, 16), (12, 32), (20, 64), (32, 96), (64, 128))),
    "d3": DetectorConfig(
        "fom", "d3", "D³", checkpoints=(
            Checkpoint(CKPT_ROOT / "D3/ckpt/classifier.pth"),
            Checkpoint(
                CKPT_ROOT / "D3/clip/ViT-L-14.pt",
                "https://openaipublic.azureedge.net/clip/models/"
                "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            ),
        ), batch_size=32, implemented=True,
        checkpoint_variants={
            "d3_clip_vitl14_shuffle14": (
                Checkpoint(CKPT_ROOT / "D3/ckpt/classifier.pth"),
                Checkpoint(
                    CKPT_ROOT / "D3/clip/ViT-L-14.pt",
                    "https://openaipublic.azureedge.net/clip/models/"
                    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
                ),
            ),
        }, default_variant="d3_clip_vitl14_shuffle14", adapter_class="D3",
        batch_size_by_free_vram_gb=((0, 8), (8, 16), (12, 32), (20, 64), (32, 96), (64, 128)),
    ),
    "de_fake": DetectorConfig("fom", "de_fake", "De-Fake", checkpoints=(
        # Upstream README labels the first file encoder and the second classifier.
        # Adapter ordering is classifier, fine-tuned CLIP, then BLIP.
        _gdrive_ckpt("De-Fake/clip_linear.pt", "1SuenxJP10VwArC6zW0SHMUGObMRqQhBD"),
        _gdrive_ckpt("De-Fake/finetune_clip.pt", "1qI7x5iodaCFq0S61LKw4wWjql7cYou_4"),
        Checkpoint(
            CKPT_ROOT / "De-Fake/model_base_capfilt_large.pth",
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/"
            "model_base_capfilt_large.pth",
        ),
    ), batch_size=1, implemented=True, adapter_class="DeFake",
        batch_size_by_free_vram_gb=((0, 1), (8, 2), (12, 4), (20, 8), (32, 16), (64, 32))),
    "dfd_fcg": DetectorConfig(
        "fom", "dfd_fcg", "DFD-FCG", "video", (
            Checkpoint(CKPT_ROOT / "DFD-FCG/checkpoint/setting.yaml"),
            _gdrive_ckpt("DFD-FCG/checkpoint/weights.ckpt", "1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk"),
            Checkpoint(
                PROJECT_ROOT / "fakeparts/detection/fom/dfd_fcg/upstream/misc/"
                "L14_real_semantic_patches_v4_2000.pickle"
            ),
            Checkpoint(
                CKPT_ROOT / "DFD-FCG/clip/ViT-L-14.pt",
                "https://openaipublic.azureedge.net/clip/models/"
                "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            ),
            Checkpoint(
                CKPT_ROOT / "DFD-FCG/face_alignment/checkpoints/2DFAN4-11f355bf06.pth.tar",
                "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
            ),
            Checkpoint(
                CKPT_ROOT / "DFD-FCG/face_alignment/s3fd-619a316812.pth",
                "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            ),
        ), batch_size=8, implemented=True, batch_unit="clips",
        checkpoint_variants={
            "full_ffg_l14": (
                Checkpoint(CKPT_ROOT / "DFD-FCG/checkpoint/setting.yaml"),
                _gdrive_ckpt("DFD-FCG/checkpoint/weights.ckpt", "1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk"),
                Checkpoint(
                    PROJECT_ROOT / "fakeparts/detection/fom/dfd_fcg/upstream/misc/"
                    "L14_real_semantic_patches_v4_2000.pickle"
                ),
                Checkpoint(
                    CKPT_ROOT / "DFD-FCG/clip/ViT-L-14.pt",
                    "https://openaipublic.azureedge.net/clip/models/"
                    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
                ),
                Checkpoint(
                    CKPT_ROOT / "DFD-FCG/face_alignment/checkpoints/2DFAN4-11f355bf06.pth.tar",
                    "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
                ),
                Checkpoint(
                    CKPT_ROOT / "DFD-FCG/face_alignment/s3fd-619a316812.pth",
                    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                ),
            ),
        }, default_variant="full_ffg_l14", adapter_class="DFDFCG",
        batch_size_by_free_vram_gb=((0, 2), (8, 8), (12, 16), (20, 24), (32, 32), (64, 48)),
    ),
    "fatformer": DetectorConfig(
        "fom", "fatformer", "FatFormer", checkpoints=(
            _gdrive_ckpt("FatFormer/checkpoint.pth", "1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj"),
            Checkpoint(
                CKPT_ROOT / "FatFormer/clip/ViT-L-14.pt",
                "https://openaipublic.azureedge.net/clip/models/"
                "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            ),
        ), batch_size=1, implemented=True, batch_unit="frames",
        checkpoint_variants={
            "fatformer_4class_vitl14": (
                _gdrive_ckpt("FatFormer/checkpoint.pth", "1Q_Kgq4ygDf8XEHgAf-SgDN6Ru_IOTLkj"),
                Checkpoint(
                    CKPT_ROOT / "FatFormer/clip/ViT-L-14.pt",
                    "https://openaipublic.azureedge.net/clip/models/"
                    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
                ),
            ),
        }, default_variant="fatformer_4class_vitl14",
        batch_size_by_free_vram_gb=((0, 1), (8, 2), (12, 4), (20, 8), (32, 16), (64, 24)),
    ),
    "univfd": DetectorConfig("fom", "univfd", "UniversalFakeDetect", checkpoints=_ckpts(
        "UniversalFakeDetect/pretrained_weights/fc_weights.pth",
        "D3/clip/ViT-L-14.pt"), implemented=True, adapter_class="UnivFD",
        batch_size_by_free_vram_gb=((0, 8), (8, 16), (12, 32), (20, 64), (32, 96), (64, 128))),
    "afprompt": DetectorConfig(
        "vlm", "afprompt", "AntifakePrompt", checkpoints=(
            _hf_ckpt("AntiFakePrompt/ckpt/COCO_150k_SD3_SD2IP.pth",
                     "OrkaZeta/AntifakePrompt", "ckpt/COCO_150k_SD3_SD2IP.pth",
                     gdrive_id="1EUnVG4OZZPXeOyWaa5P590yCKGH-nunQ"),
            _AFPROMPT_BLIP2_CKPT, _AFPROMPT_VICUNA_CKPT,
        ), batch_size=1, implemented=True, batch_unit="frames", adapter_class="AFPrompt",
        checkpoint_variants={
            "coco150k_sd3_sd2ip": (
                _hf_ckpt("AntiFakePrompt/ckpt/COCO_150k_SD3_SD2IP.pth",
                         "OrkaZeta/AntifakePrompt", "ckpt/COCO_150k_SD3_SD2IP.pth",
                         gdrive_id="1EUnVG4OZZPXeOyWaa5P590yCKGH-nunQ"),
                _AFPROMPT_BLIP2_CKPT, _AFPROMPT_VICUNA_CKPT,
            ),
            "coco180k_sd3_sd2ip_lama": (
                _hf_ckpt("AntiFakePrompt/ckpt/COCO_150k_SD3_SD2IP_lama.pth",
                         "OrkaZeta/AntifakePrompt", "ckpt/COCO_150k_SD3_SD2IP_lama.pth",
                         gdrive_id="1qnZfCknNHgC-Nhlwbab9Jg3x9sOof3gG"),
                _AFPROMPT_BLIP2_CKPT, _AFPROMPT_VICUNA_CKPT,
            ),
        }, default_variant="coco150k_sd3_sd2ip",
        batch_size_by_free_vram_gb=((0, 1), (12, 2), (20, 4), (32, 6), (64, 8)),
    ),
    "busterx": DetectorConfig(
        "vlm", "busterx", "BusterX", "video", (
            Checkpoint(CKPT_ROOT / "BusterX", hf_repo="l8cv/BusterX-plusplus",
                       required_files=("config.json", "model.safetensors.index.json",
                                       "model-00001-of-00003.safetensors",
                                       "model-00002-of-00003.safetensors",
                                       "model-00003-of-00003.safetensors",
                                       "preprocessor_config.json", "tokenizer.json",
                                       "tokenizer_config.json", "chat_template.jinja")),
        ), batch_size=1, implemented=True, batch_unit="videos",
        checkpoint_variants={
            "busterx_plusplus_qwen35": (
                Checkpoint(CKPT_ROOT / "BusterX", hf_repo="l8cv/BusterX-plusplus",
                           required_files=("config.json", "model.safetensors.index.json",
                                           "model-00001-of-00003.safetensors",
                                           "model-00002-of-00003.safetensors",
                                           "model-00003-of-00003.safetensors",
                                           "preprocessor_config.json", "tokenizer.json",
                                           "tokenizer_config.json", "chat_template.jinja")),
            ),
        }, default_variant="busterx_plusplus_qwen35",
    ),
    "fakevlm": DetectorConfig("vlm", "fakevlm", "FakeVLM", checkpoints=(
        Checkpoint(CKPT_ROOT / "FakeVLM", hf_repo="lingcco/fakeVLM",
                   required_files=("config.json", "model.safetensors.index.json",
                                   "model-00001-of-00003.safetensors",
                                   "model-00002-of-00003.safetensors",
                                   "model-00003-of-00003.safetensors",
                                   "preprocessor_config.json", "processor_config.json",
                                   "tokenizer.json", "tokenizer_config.json")),),
        batch_size=1, implemented=True),
    "mm_det": DetectorConfig("vlm", "mm_det", "MM-Det", "video", (
        _hf_ckpt("MM-Det/MM-Det/current_model.pth", "OrkaZeta/MMDet",
                 "MM-Det/current_model.pth", gdrive_id="1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy",
                 gdrive_filename="current_model.pth"),
        _hf_ckpt("MM-Det/vqvae/model.pt", "OrkaZeta/MMDet", "vqvae/model.pt",
                 gdrive_id="1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy", gdrive_filename="model.pt"),
    ), implemented=True, batch_unit="videos", adapter_class="MMDet",
        checkpoint_variants={"mmdet_llava15_7b": (
            _hf_ckpt("MM-Det/MM-Det/current_model.pth", "OrkaZeta/MMDet",
                     "MM-Det/current_model.pth", gdrive_id="1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy",
                     gdrive_filename="current_model.pth"),
            _hf_ckpt("MM-Det/vqvae/model.pt", "OrkaZeta/MMDet", "vqvae/model.pt",
                     gdrive_id="1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy", gdrive_filename="model.pt"),
        )},
        default_variant="mmdet_llava15_7b"),
    "sida": DetectorConfig("vlm", "sida", "SIDA", checkpoints=_ckpts(
        "SIDA-7B", "segment_anything/sam_vit_h_4b8939.pth"), implemented=True,
        checkpoint_variants={
            "sida_7b": (Checkpoint(CKPT_ROOT / "SIDA-7B", hf_repo="saberzl/SIDA-7B",
                required_files=("config.json", "pytorch_model.bin.index.json",
                                "pytorch_model-00001-of-00002.bin",
                                "pytorch_model-00002-of-00002.bin", "tokenizer.model",
                                "tokenizer_config.json", "special_tokens_map.json")),
                Checkpoint(CKPT_ROOT / "segment_anything/sam_vit_h_4b8939.pth",
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")),
            "sida_13b": (Checkpoint(CKPT_ROOT / "SIDA-13B", hf_repo="saberzl/SIDA-13B",
                required_files=("config.json", "pytorch_model.bin.index.json",
                                "pytorch_model-00001-of-00003.bin",
                                "pytorch_model-00002-of-00003.bin",
                                "pytorch_model-00003-of-00003.bin", "tokenizer.model",
                                "tokenizer_config.json", "special_tokens_map.json")),
                Checkpoint(CKPT_ROOT / "segment_anything/sam_vit_h_4b8939.pth",
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")),
        }, default_variant="sida_7b"),
    "drct": DetectorConfig(
        "dib", "drct", "DRCT", checkpoints=(_hf_ckpt(
            "DRCT/pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth",
            "OrkaZeta/DRCT",
            "DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"),),
        batch_size=64, implemented=True,
        batch_size_by_free_vram_gb=((0, 8), (8, 16), (12, 32), (20, 64), (32, 96), (64, 128)),
        checkpoint_variants={
            "drct2m_sdv14_convnext": (_hf_ckpt(
                "DRCT/pretrained/DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth",
                "OrkaZeta/DRCT", "DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"),),
            "drct2m_sdv2_convnext": (_hf_ckpt(
                "DRCT/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth",
                "OrkaZeta/DRCT", "DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth"),),
            "genimage_sdv14_convnext": (_hf_ckpt(
                "DRCT/pretrained/GenImage/sdv14/convnext_base_in22k_224_drct_amp_crop/last_acc0.9991.pth",
                "OrkaZeta/DRCT", "GenImage/sdv14/convnext_base_in22k_224_drct_amp_crop/last_acc0.9991.pth"),),
            "drct2m_sdv14_clip_vitl14": (_hf_ckpt(
                "DRCT/pretrained/DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth",
                "OrkaZeta/DRCT", "DRCT-2M/sdv14/clip-ViT-L-14_224_drct_amp_crop/13_acc0.9664.pth"),),
            "drct2m_sdv2_clip_vitl14": (_hf_ckpt(
                "DRCT/pretrained/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth",
                "OrkaZeta/DRCT", "DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth"),),
            "genimage_sdv14_clip_vitl14": (_hf_ckpt(
                "DRCT/pretrained/GenImage/sdv14/clip-ViT-L-14_224_drct_amp_crop/2_acc0.9558.pth",
                "OrkaZeta/DRCT", "GenImage/sdv14/clip-ViT-L-14_224_drct_amp_crop/2_acc0.9558.pth"),),
        }, default_variant="drct2m_sdv14_convnext",
    ),
    "nsg_vd": DetectorConfig("dib", "nsg_vd", "NSG-VD", "video",
        _nsg_ckpts("NSG-VD/ckpts/standard-Pika-mp.pth"), implemented=True,
        checkpoint_variants={
            **{f"{task}_{generator.lower()}_{kind}": _nsg_ckpts(
                f"NSG-VD/ckpts/{task}-{generator}-{kind}.pth")
              for task, generator, kind in (
                ("standard", "Pika", "d"), ("standard", "Pika", "mp"),
                ("standard", "SEINE", "d"), ("standard", "SEINE", "mp"),
                ("unbalance", "SEINE", "d"), ("unbalance", "SEINE", "mp"))},
            **{f"baseline_{task}_{generator.lower()}_{family}": (
                Checkpoint(CKPT_ROOT / f"NSG-VD/baselines/{task}-{generator}-{family}.pth"),)
               for task in ("standard", "unbalance")
               for generator in ("Pika", "SEINE")
               for family in ("demamba", "npr", "stil", "tall")},
        }, default_variant="standard_pika_mp", adapter_class="NSGVD"),
}

for _detector_id, _detector_config in DET_MODELS.items():
    object.__setattr__(_detector_config, "_registry_id", _detector_id)

# Public detector IDs follow results/Stats.md. Accept legacy spellings at CLI
# boundaries, but always write paths and result records with the canonical ID.
DETECTOR_ID_ALIASES = {
    "de-fake": "de_fake",
    "dfd-fcg": "dfd_fcg",
    "mm-det": "mm_det",
    "nsg-vd": "nsg_vd",
}


def canonical_detector_id(value: str) -> str:
    detector_id = value.strip().lower()
    detector_id = DETECTOR_ID_ALIASES.get(detector_id, detector_id)
    if detector_id not in DET_MODELS:
        raise KeyError(detector_id)
    return detector_id


def resolve_checkpoints(config: DetectorConfig, variant: str | None = None) -> tuple[str, tuple[Checkpoint, ...]]:
    """Return the explicit checkpoint set selected for one detector version."""
    selected = variant or config.default_variant
    if config.checkpoint_variants:
        if selected is None:
            raise ValueError(
                f"{config.class_name} requires --version; available: "
                f"{', '.join(sorted(config.checkpoint_variants))}"
            )
        # Canonical version IDs use underscores. Resolve older hyphenated
        # spellings at the CLI boundary, but return the underscore ID so all
        # output filenames and summaries remain canonical.
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", selected).strip("_")
        matches = {key.lower(): key for key in config.checkpoint_variants}
        canonical = matches.get(normalized.lower())
        try:
            if canonical is None:
                raise KeyError(selected)
            return canonical, config.checkpoint_variants[canonical]
        except KeyError as exc:
            choices = ", ".join(sorted(config.checkpoint_variants))
            raise ValueError(f"Unknown {config.class_name} version {selected!r}; available: {choices}") from exc
    if selected not in (None, "default"):
        raise ValueError(f"{config.class_name} has no selectable versions")
    return "default", config.checkpoints
