from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None

# Split-mode globals (target and draft on disjoint GPUs)
_SPLIT_MODE = False
_TARGET_RANKS = []
_DRAFT_RANKS = []
_DRAFT_FSDP_GROUP = None
_DRAFT_DP_GROUP_SPLIT = None
_TRANSFER_GROUP = None


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH


def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def is_split_mode():
    global _SPLIT_MODE
    return _SPLIT_MODE


def is_target_rank():
    global _SPLIT_MODE, _TARGET_RANKS
    if not _SPLIT_MODE:
        return False
    return dist.get_rank() in _TARGET_RANKS


def is_draft_rank():
    global _SPLIT_MODE, _DRAFT_RANKS
    if not _SPLIT_MODE:
        return False
    return dist.get_rank() in _DRAFT_RANKS


def get_draft_fsdp_group():
    global _DRAFT_FSDP_GROUP
    return _DRAFT_FSDP_GROUP


def get_draft_dp_group_split():
    global _DRAFT_DP_GROUP_SPLIT
    return _DRAFT_DP_GROUP_SPLIT


def get_transfer_group():
    global _TRANSFER_GROUP
    return _TRANSFER_GROUP


def get_transfer_src_rank():
    """Return the global rank of the source for data transfer (target rank 0)."""
    return 0


def init_distributed_split(timeout: int = 10, tp_size: int = 1, draft_dp_size: int = 1):
    """Initialize distributed training with split GPU allocation.

    Target model gets GPUs [0, tp_size-1] for TP inference.
    Draft model gets GPUs [tp_size, world_size-1] for FSDP training + DP.

    Args:
        timeout: Timeout for collective communication in minutes.
        tp_size: Number of GPUs for target model (tensor parallelism degree).
        draft_dp_size: Number of data-parallel replicas on draft side.
            draft_gpu_count / draft_dp_size = FSDP group size per replica.
    """
    global _SPLIT_MODE, _TARGET_RANKS, _DRAFT_RANKS
    global _TP_GROUP, _DP_GROUP, _TP_DEVICE_MESH, _DP_DEVICE_MESH
    global _DRAFT_FSDP_GROUP, _DRAFT_DP_GROUP_SPLIT, _TRANSFER_GROUP

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size > tp_size, (
        f"Split mode requires world_size ({world_size}) > tp_size ({tp_size}). "
        f"Need at least 1 GPU for draft model."
    )

    draft_gpu_count = world_size - tp_size
    assert draft_gpu_count % draft_dp_size == 0, (
        f"Number of draft GPUs ({draft_gpu_count}) must be divisible by "
        f"draft_dp_size ({draft_dp_size})."
    )
    fsdp_group_size = draft_gpu_count // draft_dp_size

    _SPLIT_MODE = True
    _TARGET_RANKS = list(range(tp_size))
    _DRAFT_RANKS = list(range(tp_size, world_size))

    # Create target TP group (all ranks must participate in new_group)
    target_group = dist.new_group(_TARGET_RANKS)

    # Create per-DP-replica FSDP groups on draft side
    # E.g., 4 draft GPUs [4,5,6,7], draft_dp_size=2, fsdp_size=2:
    #   FSDP group 0: [4,5], FSDP group 1: [6,7]
    my_fsdp_group = None
    for i in range(draft_dp_size):
        start = tp_size + i * fsdp_group_size
        fsdp_ranks = list(range(start, start + fsdp_group_size))
        group = dist.new_group(fsdp_ranks)
        if rank in fsdp_ranks:
            my_fsdp_group = group

    # Create DP groups across FSDP replicas (ranks at same position within each replica)
    # E.g., draft ranks [4,5,6,7], draft_dp_size=2, fsdp_size=2:
    #   DP group 0: [4,6] (position 0 in each replica)
    #   DP group 1: [5,7] (position 1 in each replica)
    my_dp_group = None
    for pos in range(fsdp_group_size):
        dp_ranks = [tp_size + i * fsdp_group_size + pos for i in range(draft_dp_size)]
        group = dist.new_group(dp_ranks)
        if rank in dp_ranks:
            my_dp_group = group

    # Transfer group: target rank 0 + all draft ranks (for broadcasting target output)
    transfer_group = dist.new_group([0] + _DRAFT_RANKS)

    # Create single-rank groups for SP (split mode doesn't use sequence parallelism)
    # Needed by DataCollatorWithPadding which calls get_draft_sp_group()
    for r in range(world_size):
        group = dist.new_group([r])
        if rank == r:
            my_sp_group = group

    global _DRAFT_SP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP
    _DRAFT_DP_GROUP = my_dp_group if rank in _DRAFT_RANKS else None
    _DRAFT_SP_GROUP = my_sp_group
    _SP_ULYSSES_GROUP = my_sp_group
    _SP_RING_GROUP = my_sp_group

    # Set globals
    _TP_GROUP = target_group
    # DeviceMesh constructor is collective — all ranks must call it.
    # Create a 1D mesh over target ranks only. Draft ranks participate in the
    # collective but never use the resulting mesh.
    _TP_DEVICE_MESH = dist.DeviceMesh(
        "cuda", torch.tensor(_TARGET_RANKS, dtype=torch.int)
    )
    _DRAFT_FSDP_GROUP = my_fsdp_group
    _DRAFT_DP_GROUP_SPLIT = my_dp_group
    _TRANSFER_GROUP = transfer_group

    # In split mode, _DP_GROUP is only used by DataLoader's DistributedSampler.
    # All ranks use single-rank group → num_replicas=1 → all get same full dataset.
    # - Target ranks: required for TP (all TP ranks must process same input)
    # - Draft ranks: ensures step count matches target (draft receives data via transfer)
    _DP_GROUP = my_sp_group

    print_with_rank(
        f"split mode: {'target' if rank in _TARGET_RANKS else 'draft'} rank, "
        f"tp_size={tp_size}, draft_gpu_count={draft_gpu_count}, "
        f"draft_dp_size={draft_dp_size}, fsdp_group_size={fsdp_group_size}"
    )


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert (
        world_size == tp_size * dp_size
    ), f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size*sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DP_GROUP = dp_group
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_SP_GROUP = draft_device_mesh.get_group("sp")
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DRAFT_DP_GROUP
    global _SPLIT_MODE, _DRAFT_FSDP_GROUP, _DRAFT_DP_GROUP_SPLIT, _TRANSFER_GROUP
    if _SPLIT_MODE:
        # In split mode, clean up split-specific groups
        dist.destroy_process_group()
    else:
        dist.destroy_process_group(_TP_GROUP)
        dist.destroy_process_group(_DP_GROUP)
        dist.destroy_process_group(_SP_ULYSSES_GROUP)
        dist.destroy_process_group(_SP_RING_GROUP)
        dist.destroy_process_group(_DRAFT_DP_GROUP)
        dist.destroy_process_group(_DRAFT_SP_GROUP)
        dist.destroy_process_group()


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gather_tensor = torch.cat(obj_list, dim=dim)
    return gather_tensor


def all_gather_tensor(
    local_tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


# Adapted from https://github.com/volcengine/verl/blob/a0e8e4472b8b472409defb0c8fcc5162301450af/verl/utils/ulysses.py#L194
class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outputs_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    if not group:
        group = get_draft_sp_group()
    if torch.distributed.get_world_size(group) == 1:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    return x


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
