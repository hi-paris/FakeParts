from importlib import import_module

_CLASS_MODULES = {
# OoB Models
    "AIGVDet": "oob.aigvdet.detector",
    "CNNDet": "oob.cnndet.detector",
    "HiFiNet": "oob.hifi.detector",
    "IMLViT": "oob.iml_vit.detector",
    "NPR": "oob.npr.detector",
    "TALL": "oob.tall.detector",
    "TFCU": "oob.tfcu.detector",

# FoM models
    "C2P": "fom.c2p.detector",
    "D3": "fom.d3.detector",
    "DeFake": "fom.de_fake.detector",
    "DFDFCG": "fom.dfd_fcg.detector",
    "FatFormer": "fom.fatformer.detector",
    "UnivFD": "fom.univfd.detector",

# VLM models
    "AFPrompt": "vlm.afprompt.detector",
    "BusterX": "vlm.busterx.detector",
    "FakeVLM": "vlm.fakevlm.detector",
    "MMDet": "vlm.mm_det.detector",
    "SIDA": "vlm.sida.detector",

# DiB models
    "DRCT": "dib.drct.detector",
    "NSGVD": "dib.nsg_vd.detector",

}

__all__ = [
    # OoB Detector
    "AIGVDet",
    "CNNDet",
    "HiFiNet",
    "IMLViT",
    "NPR",
    "TALL",
    "TFCU",
    # FoM Models
    "C2P",
    "D3",
    "DeFake",
    "DFDFCG",
    "FatFormer",
    "UnivFD",
    # VLM Models
    "AFPrompt",
    "BusterX",
    "FakeVLM",
    "MMDet",
    "SIDA",
    # DiB Models
    "DRCT",
    "NSGVD",
]


def __getattr__(name):
    if name not in _CLASS_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    detector = getattr(import_module(f".{_CLASS_MODULES[name]}", __name__), name)
    globals()[name] = detector
    return detector
