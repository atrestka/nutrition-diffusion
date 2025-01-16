# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Food-101."""

from diffusion.datasets.food101.food101 import StreamingFood101Dataset, build_streaming_Food101_dataloader

__all__ = [
    'build_streaming_Food101_dataloader',
    'StreamingFood101Dataset',
]
