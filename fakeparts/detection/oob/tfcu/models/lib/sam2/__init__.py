# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import sam2_configs  # type: ignore[import-not-found]
except ImportError:
    # TFCU uses the bundled SAM2 layers directly and does not load SAM2 Hydra
    # configs. Keep importing those layers possible when the optional config
    # package is absent from this upstream snapshot.
    pass
else:
    from hydra import initialize_config_module

    initialize_config_module("sam2_configs", version_base="1.2")
