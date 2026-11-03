"""Canonical display names and IDs used by the dataset and paper outputs."""
from __future__ import annotations

import re

TASKS = {
    "t2v": ("Text-to-Video", "T2V"),
    "it2v": ("Image-Text-to-Video", "IT2V"),
    "style": ("StyleTransfer", "Style"),
    "faceswap": ("FaceSwap", "FaceSwap"),
    "inpainting": ("Inpainting", "Inpainting"),
    "outpainting": ("Outpainting", "Outpainting"),
    "interpolation": ("Interpolation", "Interpolation"),
    "extrapolation": ("Extrapolation", "Extrapolation"),
    "real": ("Real", "Real"),
}

FAMILY_ORDER = ("oob", "fom", "vlm", "dib")
DETECTOR_ORDER = (
    "cnndet", "hifi", "tall", "npr", "aigvdet", "iml_vit", "tfcu",
    "de_fake", "univfd", "fatformer", "d3", "c2p", "dfd_fcg", "mm_det",
    "fakevlm", "sida", "afprompt", "busterx", "drct", "nsg_vd",
)
TASK_ORDER = (
    "t2v", "it2v", "style", "faceswap", "inpainting", "outpainting",
    "interpolation", "extrapolation", "real",
)
GENERATOR_ORDER = {
    "t2v": (
        "sora_1080p", "sora_720p", "opensora", "cogvidx", "allegroai",
        "mochi1", "ltx_video", "hunyuanvid", "veo2", "wan21",
    ),
    "it2v": ("opensora", "wan21"),
    "style": ("rave", "anyv2v"),
    "faceswap": ("insightface", "liveportrait", "hunyuanportrait"),
    "inpainting": ("propainter", "rovi", "diffueraser"),
    "outpainting": ("akira",),
    "interpolation": ("framer",),
    "extrapolation": ("cosmos2",),
}


def table_order_key(task_id: str, generator_id: str = "") -> tuple[int, int, str]:
    """Sort a task/method pair in the paper-table order, then alphabetically if unknown."""
    task = canonical_task_id(task_id)
    generator = canonical_generator_id(generator_id)
    task_index = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
    methods = GENERATOR_ORDER.get(task, ())
    method_index = methods.index(generator) if generator in methods else len(methods)
    return task_index, method_index, generator.casefold()

GENERATORS = {
    "veo2": ("Veo 2 (720p)", "Veo2 (720p)"), "wan21": ("Wan 2.1", "Wan2.1"),
    "sora_720p": ("Sora (720p)", "Sora (720p)"),
    "sora_1080p": ("Sora (1080p)", "Sora (1080p)"),
    "opensora": ("OpenSora", "OpenSora"), "mochi1": ("Mochi-1", "Mochi1"),
    "ltx_video": ("LTX-Video", "LTXVid"), "hunyuanvid": ("HunyuanVideo", "HunyuanVid"),
    "cogvidx": ("CogVideoX", "CogVidX"), "allegroai": ("AllegroAI", "AllegroAI"),
    "rave": ("RAVE", "RAVE"), "anyv2v": ("AnyV2V", "AnyV2V"),
    "akira": ("Akira", "Akira"), "framer": ("Framer", "Framer"),
    "rovi": ("ROVI", "ROVI"), "propainter": ("ProPainter", "ProPainter"),
    "diffueraser": ("DiffuEraser", "DiffuEraser"), "liveportrait": ("LivePortrait", "LivePortrait"),
    "insightface": ("InsightFace", "InsightFace"), "hunyuanportrait": ("HunyuanPortrait", "HunyuanPortrait"),
    # Extrapolation
    "cosmos2": ("Cosmos-Predict2", "Cosmos2"),
    # Real reference datasets (stored in the same method columns in tables).
    "tenk_youtube": ("TenK Real YouTube", "TenkYouTube"),
    "youtube_vos_2019": ("YouTube VOS 2019", "YouTubeVOS2019"),
}

FAMILIES = {
    "oob": ("Out-of-the-Box", "OoB"),
    "fom": ("Foundation Models", "FoM"),
    "vlm": ("Vision-Language Models", "VLM"),
    "dib": ("Diffusion-Based", "DiB"),
}

DETECTORS = {
    "aigvdet": ("AIGVDet", "AIGV"), "cnndet": ("CNNDetection", "CNNDet"),
    "hifi": ("HiFi-IFDL", "HiFi"), "iml_vit": ("IML-ViT", "IML-ViT"),
    "npr": ("NPR", "NPR"), "tall": ("TALL", "TALL"), "tfcu": ("TFCU", "TFCU"),
    "c2p": ("C2P", "C2P"), "d3": ("D³", "D3"), "de_fake": ("De-Fake", "De-Fake"),
    "dfd_fcg": ("DFD-FCG", "DFD-FCG"), "fatformer": ("FatFormer", "Fatfrm."),
    "univfd": ("UniversalFakeDetect", "UnivFD"), "afprompt": ("AntiFakePrompt", "AFPrompt"),
    "busterx": ("BusterX", "BusterX"), "fakevlm": ("FakeVLM", "FakeVLM"),
    "mm_det": ("MM-Det", "MM-Det"), "sida": ("SIDA", "SIDA"),
    "drct": ("DRCT", "DRCT"), "nsg_vd": ("NSG-VD", "NSG-VD"),
}

# Historical directory/metadata spellings accepted at boundaries and migrated in data.
GENERATOR_ALIASES = {
    "veo_2": "veo2", "veo2": "veo2", "allegro": "allegroai", "cogvideox": "cogvidx",
    "hunyuanvideo": "hunyuanvid", "mochi": "mochi1", "cosmos_predict2": "cosmos2",
}
TASK_ALIASES = {"image_text_to_video": "it2v", "text_to_video": "t2v", "face_swap": "faceswap", "style_transfer": "style"}

def canonical_generator_id(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    return GENERATOR_ALIASES.get(key, key)

def canonical_task_id(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    return TASK_ALIASES.get(key, key)

def version_name(detector_id: str, version_id: str) -> str:
    """Human-readable version labels; unlisted IDs are title-cased safely."""
    labels = {
        "default": "Default", "blur_jpg_prob0_1": "Blur JPEG, p=0.1",
        "blur_jpg_prob0_5": "Blur JPEG, p=0.5", "iml_vit_release": "Release",
        "npr_release": "Release", "d3_clip_vitl14_shuffle14": "CLIP ViT-L/14, shuffle 14",
        "full_ffg_l14": "Full FFG L/14", "fatformer_4class_vitl14": "Four-class ViT-L/14",
        "coco150k_sd3_sd2ip": "COCO 150K SD3+SD2IP",
        "coco180k_sd3_sd2ip_lama": "COCO 180K SD3+SD2IP+LaMa",
        "busterx_plusplus_qwen35": "BusterX++ Qwen 3.5", "mmdet_llava15_7b": "LLaVA 1.5 7B",
        "sida_7b": "7B", "sida_13b": "13B",
        "drct2m_sdv14_convnext": "DRCT-2M SD v1.4 ConvNeXt",
        "drct2m_sdv2_convnext": "DRCT-2M SD v2.0 ConvNeXt",
        "genimage_sdv14_convnext": "GenImage SD v1.4 ConvNeXt",
        "drct2m_sdv14_clip_vitl14": "DRCT-2M SD v1.4 CLIP ViT-L/14",
        "drct2m_sdv2_clip_vitl14": "DRCT-2M SD v2.0 CLIP ViT-L/14",
        "genimage_sdv14_clip_vitl14": "GenImage SD v1.4 CLIP ViT-L/14",
    }
    if version_id in labels:
        return labels[version_id]
    nsg = re.fullmatch(r"(baseline_)?(standard|unbalance)_(pika|seine)_(d|mp|demamba|npr|stil|tall)", version_id)
    if nsg:
        baseline, balance, generator, variant = nsg.groups()
        generator_name = {"pika": "Pika", "seine": "SEINE"}[generator]
        title = f"{balance.title()} {generator_name}"
        if baseline:
            classifier = {"demamba": "DeMamba", "npr": "NPR", "stil": "STIL", "tall": "TALL"}[variant]
            return f"{title} {classifier} baseline"
        return f"{title} dMMD ({variant})"
    return version_id.replace("_", " ").title()
