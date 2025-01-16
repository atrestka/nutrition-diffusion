# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming food-101 dataset."""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader


class StreamingFood101Dataset(StreamingDataset):
    """Implementation of the LAION dataset as a streaming dataset.

    Args:
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from. StreamingLAIONDataset
            uses either ``streams`` or ``remote``/``local``. Default:``None``.
        remote (str, optional): Remote directory (S3 or local filesystem) where dataset is stored. Default: ``None``.
        local (str, optional): Local filesystem directory where dataset is cached during operation. Default: ``None``.
        split (str, optional): The dataset split to use. Currently, only ``None`` is supported. Default: ``None``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``False``.
        shuffle_algo (str): What shuffle algorithm to use. Default: ``'py1s'``.
        shuffle_block_size (int): Unit of shuffling. Default: ``1 << 18``.
        predownload (Optional[int]): The number of samples to prefetch. If ``None``, its value is set to ``8 * batch_size``. Default: ``None``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
    """

    def __init__(
        self,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1s',
        shuffle_block_size: int = 1 << 18,
        predownload: Optional[int] = None,
        download_retry: int = 2,
        download_timeout: float = 120,
        batch_size: Optional[int] = None,
        num_canonical_nodes: Optional[int] = None,
    ) -> None:

        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_block_size=shuffle_block_size,
            predownload=predownload,
            keep_zip=False,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=None,
            batch_size=batch_size,
            num_canonical_nodes=num_canonical_nodes,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        out = {}
        if 'caption_latents' in sample:
            out['caption_latents'] = torch.from_numpy(
                np.frombuffer(sample['caption_latents'], dtype=np.float16).copy()).reshape(77, 1024)
        if 'latents_256' in sample:
            out['image_latents'] = torch.from_numpy(np.frombuffer(sample['latents_256'],
                                                                  dtype=np.float16).copy()).reshape(4, 32, 32)
        return out


def build_streaming_Food101_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    num_samples: Optional[int] = None,
    predownload: int = 100_000,
    download_retry: int = 2,
    download_timeout: float = 120,
    drop_last: bool = True,
    shuffle: bool = True,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs,
):
    """Builds a streaming food-101 dataloader.

    Args:
        remote (str, Sequence[str]): One or more remote directories (S3 or local filesystem) where dataset is stored.
        local (str, Sequence[str]): One or more local filesystem directories where dataset is cached during operation.
        batch_size (int): The batch size to use.
        num_samples (Optional[int]): The number of samples to use. Default: ``None`` uses all available samples.
        predownload (Optional[int]): The number of samples to prefetch. Default: ``100_000``.
        download_retry (Optional[int]): The number of times to retry a download. Default: ``2``.
        download_timeout (Optional[float]): The timeout for a download. Default: ``120``.
        drop_last (bool): Whether to drop the last batch if it is incomplete. Default: ``True``.
        shuffle (bool): Whether to shuffle the samples in this dataset. Default: ``True``.
        num_canonical_nodes (int, optional): The number of canonical nodes for shuffle. Default: ``None``.
        **dataloader_kwargs: Additional arguments to pass to the dataloader.
    """
    if isinstance(remote, str) and isinstance(local, str):
        # Hacky... make remote and local lists to simplify downstream code
        remote, local = [remote], [local]
    elif isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f'remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}')
    else:
        ValueError(f'remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.')

    # Create a Stream for each (remote, local) pair
    streams = []
    for r, l in zip(remote, local):
        streams.append(Stream(remote=r, local=l, download_retry=download_retry, download_timeout=download_timeout))

    dataset = StreamingFood101Dataset(
        streams=streams,
        split=None,
        shuffle=shuffle,
        predownload=predownload,
        download_retry=download_retry,
        download_timeout=download_timeout,
        batch_size=batch_size,
        num_canonical_nodes=num_canonical_nodes,
    )
    # Create a subset of the dataset
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))  # type: ignore

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
