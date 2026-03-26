"""Data transfer utilities for split-GPU training mode.

Handles broadcasting Eagle3TargetOutput from target ranks to draft ranks.
"""

from typing import List, Tuple

import torch
import torch.distributed as dist

from specforge.distributed import (
    get_draft_dp_group_split,
    get_tp_group,
    get_transfer_group,
    get_transfer_src_rank,
)
from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

# Fields to transfer and their dtypes
_TENSOR_FIELDS: List[Tuple[str, torch.dtype]] = [
    ("hidden_states", torch.bfloat16),
    ("target", torch.bfloat16),
    ("loss_mask", torch.long),
    ("input_ids", torch.long),
    ("attention_mask", torch.long),
]


def _gather_from_tp(tensor: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
    """Gather tensor along batch dim from all TP ranks to rank 0."""
    tp_size = dist.get_world_size(tp_group)
    if tp_size <= 1:
        return tensor
    tp_rank = dist.get_rank(tp_group)

    # All TP ranks have the same data when batch_size < tp_size
    if tensor.shape[0] < tp_size:
        return tensor

    # Gather all chunks on TP rank 0
    gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=0)


def _shard_by_dp(tensor: torch.Tensor, dp_group: dist.ProcessGroup) -> torch.Tensor:
    """Shard tensor along batch dim for data-parallel replicas."""
    if dp_group is None:
        return tensor
    dp_size = dist.get_world_size(dp_group)
    if dp_size <= 1:
        return tensor
    dp_rank = dist.get_rank(dp_group)
    if tensor.shape[0] < dp_size:
        return tensor
    return tensor.chunk(dp_size, dim=0)[dp_rank]


class TargetOutputTransfer:
    """Handles transfer of Eagle3TargetOutput from target rank 0 to all draft ranks.

    Uses dist.broadcast via the transfer group (target rank 0 + all draft ranks).
    Broadcasts metadata (shapes) first, then each tensor.

    On target rank 0: call send() with the eagle3_data.
    On other target ranks: call send_noop() to participate in TP gather without broadcast.
    On draft ranks: call receive() to get the data.
    """

    def __init__(self):
        self.transfer_group = get_transfer_group()
        self.src_rank = get_transfer_src_rank()

    def send(self, eagle3_data: Eagle3TargetOutput) -> None:
        """Broadcast eagle3_data from target rank 0 to all draft ranks.

        Called only on target rank 0 (global rank 0).
        """
        # First broadcast metadata: number of fields and shape of each tensor
        for field_name, expected_dtype in _TENSOR_FIELDS:
            tensor = getattr(eagle3_data, field_name)
            # Broadcast number of dimensions and shape
            ndim = tensor.ndim
            meta = torch.tensor(
                [ndim] + list(tensor.shape), dtype=torch.long, device="cuda"
            )
            meta_size = torch.tensor([len(meta)], dtype=torch.long, device="cuda")
            dist.broadcast(meta_size, src=self.src_rank, group=self.transfer_group)
            dist.broadcast(meta, src=self.src_rank, group=self.transfer_group)
            # Broadcast the tensor itself
            dist.broadcast(
                tensor.to(expected_dtype).contiguous(),
                src=self.src_rank,
                group=self.transfer_group,
            )

    def receive(self) -> Eagle3TargetOutput:
        """Receive eagle3_data from target rank 0.

        Called on all draft ranks.
        """
        received = {}
        for field_name, expected_dtype in _TENSOR_FIELDS:
            # Receive metadata
            meta_size = torch.empty(1, dtype=torch.long, device="cuda")
            dist.broadcast(meta_size, src=self.src_rank, group=self.transfer_group)
            meta = torch.empty(meta_size.item(), dtype=torch.long, device="cuda")
            dist.broadcast(meta, src=self.src_rank, group=self.transfer_group)

            ndim = meta[0].item()
            shape = meta[1 : 1 + ndim].tolist()

            # Receive tensor
            tensor = torch.empty(shape, dtype=expected_dtype, device="cuda")
            dist.broadcast(tensor, src=self.src_rank, group=self.transfer_group)
            received[field_name] = tensor

        return Eagle3TargetOutput(**received)

    def gather_and_send(self, eagle3_data: Eagle3TargetOutput) -> None:
        """Gather full batch from TP ranks, then broadcast from rank 0.

        Called on target rank 0. Gathers across TP group before broadcasting.
        """
        tp_group = get_tp_group()
        gathered = Eagle3TargetOutput(
            hidden_states=_gather_from_tp(eagle3_data.hidden_states, tp_group),
            target=_gather_from_tp(eagle3_data.target, tp_group),
            loss_mask=_gather_from_tp(eagle3_data.loss_mask, tp_group),
            input_ids=_gather_from_tp(eagle3_data.input_ids, tp_group),
            attention_mask=_gather_from_tp(eagle3_data.attention_mask, tp_group),
        )
        self.send(gathered)

    def receive_and_shard(self) -> Eagle3TargetOutput:
        """Receive data and shard by draft DP group.

        Called on draft ranks. Each DP replica gets its 1/draft_dp_size slice.
        """
        data = self.receive()
        dp_group = get_draft_dp_group_split()
        return Eagle3TargetOutput(
            hidden_states=_shard_by_dp(data.hidden_states, dp_group),
            target=_shard_by_dp(data.target, dp_group),
            loss_mask=_shard_by_dp(data.loss_mask, dp_group),
            input_ids=_shard_by_dp(data.input_ids, dp_group),
            attention_mask=_shard_by_dp(data.attention_mask, dp_group),
        )
