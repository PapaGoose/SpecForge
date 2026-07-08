#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script.

Two execution modes:

* Colocated (default): every rank holds a TP shard of the target model plus a
  full copy of the draft model; the target forward and the draft train step
  run back-to-back on the same GPUs.

* Disaggregated (``--disagg-inference-ranks N``, sglang backend only): the
  first N ranks run the target model (TP=N) and stream hidden states over NCCL
  p2p; the remaining ranks train the draft under FSDP, one data shard each,
  and never load the target trunk. Example on 8 GPUs (4 inference + 4 train):

      torchrun --nproc_per_node 8 scripts/train_dflash.py \
          --target-model-backend sglang --disagg-inference-ranks 4 ...
"""

import argparse
import functools
import logging
import math
import os
import shutil
import time
import warnings
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel, build_target_layer_ids
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
    join_sglang_collective_init,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    get_local_device,
    load_tokenizer,
    print_on_rank0,
    print_with_rank,
)

logger = logging.getLogger(__name__)


def print_on_main(message, main_rank=0):
    """print_on_rank0, but for runs where the logging rank is not global rank 0
    (disaggregated mode logs from the first training rank)."""
    if dist.get_rank() == main_rank:
        logger.info(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables. "
        "Only applies when --loss-type dflash.",
    )
    model_group.add_argument(
        "--loss-type",
        type=str,
        default=None,
        choices=[
            "dflash",
            "vp_drafter",
            "dpace",
            "dpace-cumulative-confidence-only",
            "dpace-continuation-value-only",
        ],
        help=(
            "Training objective. If omitted, reads dflash_config.training_mode or "
            "dflash_config.loss_type from the draft config, defaulting to dflash."
        ),
    )
    model_group.add_argument(
        "--dpace-alpha",
        type=float,
        default=0.5,
        help="Smoothing alpha for D-PACE position weights.",
    )
    model_group.add_argument(
        "--prefix-weight-base",
        type=float,
        default=None,
        help=(
            "VP-Drafter prefix length sampling base. Values below 1 prefer shorter "
            "visible prefixes; defaults to dflash_config.prefix_weight_base or 0.9."
        ),
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models, "
        "'model.language_model.embed_tokens.weight' for multimodal models like Qwen3.5-A3B.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)
    dist_group.add_argument(
        "--disagg-inference-ranks",
        type=int,
        default=0,
        help="Disaggregated mode: the first N ranks run the sglang target model "
        "(TP=N) and stream hidden states to the remaining ranks, which train "
        "the draft model (FSDP, one data shard each) without loading the "
        "target trunk. 0 disables (colocated mode). Requires "
        "--target-model-backend sglang and world size divisible by N.",
    )

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_draft_config(args):
    """Load or auto-generate the draft config and resolve loss-type defaults."""
    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        # Warn if command-line args differ from config
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: checkpoint block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using checkpoint value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    args.loss_type = (
        args.loss_type
        or draft_config.dflash_config.get("training_mode")
        or draft_config.dflash_config.get("loss_type")
        or "dflash"
    )
    if args.prefix_weight_base is None:
        args.prefix_weight_base = draft_config.dflash_config.get(
            "prefix_weight_base", 0.9
        )

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")
    print_on_rank0(f"Using DFlash training loss_type: {args.loss_type}")
    return draft_config


def resolve_target_layer_ids(draft_config):
    """Mirror DFlashDraftModel's target_layer_ids resolution without
    instantiating the (GPU-sized) draft model — used by inference-only ranks."""
    dflash_config = getattr(draft_config, "dflash_config", None) or {}
    return list(
        dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(
                draft_config.num_target_layers, draft_config.num_hidden_layers
            ),
        )
    )


def build_target_model(args) -> DFlashTargetModel:
    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    device = get_local_device()
    return get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device.type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )


def build_models(args) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model = build_target_model(args)
    draft_config = build_draft_config(args)

    device = get_local_device()
    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_train_dataset(args, tokenizer):
    """Build the processed + filtered train dataset (deterministic across ranks)."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )
    return train_eagle3_dataset


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders (colocated mode)."""
    train_eagle3_dataset = build_train_dataset(args, tokenizer)

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def send_hidden_states(hidden_states: torch.Tensor, dst: int) -> None:
    """P2P-send one shard's hidden states to its training rank.

    A small shape header goes first so the receiver can detect dataloader
    desync (recv into a wrong-sized buffer would corrupt silently)."""
    header = torch.tensor(
        list(hidden_states.shape), dtype=torch.long, device=hidden_states.device
    )
    dist.send(header, dst=dst)
    dist.send(hidden_states.contiguous(), dst=dst)


def recv_hidden_states(expected_shape, device, src: int) -> torch.Tensor:
    header = torch.empty(len(expected_shape), dtype=torch.long, device=device)
    dist.recv(header, src=src)
    received = tuple(header.tolist())
    if received != tuple(expected_shape):
        raise RuntimeError(
            f"Hidden-state shape {received} from inference rank {src} does not "
            f"match the local batch {tuple(expected_shape)}; the inference and "
            "training dataloaders are out of sync."
        )
    hidden_states = torch.empty(
        *expected_shape, dtype=torch.bfloat16, device=device
    )
    dist.recv(hidden_states, src=src)
    return hidden_states


def save_checkpoint(
    args,
    epoch,
    step,
    dflash_model,
    draft_model,
    optimizer,
    save_rank=0,
    process_group=None,
):
    """Save checkpoint. In disaggregated mode only the training ranks enter
    here, so barriers are scoped to their group and the writer is the first
    training rank rather than global rank 0."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == save_rank:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier(group=process_group)

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == save_rank:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            logger.info(f"Saved checkpoint to {save_dir}")

    dist.barrier(group=process_group)


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
    main_rank: int = 0,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy

    print_on_main(
        f"{mode.capitalize()} - Step {global_step} [{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], Loss: {loss:.4f}, Acc: {accuracy:.4f}",
        main_rank=main_rank,
    )

    tracker.log(logdict, step=global_step)


def run_training(
    args,
    draft_model: DFlashDraftModel,
    tokenizer,
    train_dataloader: DataLoader,
    hidden_provider: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    draft_model_last_checkpoint: Optional[str] = None,
    ckpt_info: Tuple[int, int] = (0, 0),
    dp_group: Optional[dist.ProcessGroup] = None,
    main_rank: int = 0,
):
    """Wrap the draft in FSDP and run the train loop.

    ``hidden_provider(input_ids, attention_mask, loss_mask)`` returns the
    target hidden states for the local batch: in colocated mode it runs the
    target forward on this rank, in disaggregated mode it receives them from
    the paired inference rank. ``dp_group`` scopes FSDP and metric reduction
    (None = the whole world, i.e. colocated mode).
    """
    device = get_local_device()
    rank = dist.get_rank()

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print_on_main("Loaded draft model weights from checkpoint", main_rank)

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_main(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}",
                main_rank,
            )

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif (
        dflash_config := getattr(draft_model.config, "dflash_config", {})
    ) and dflash_config.get("mask_token_id") is not None:
        mask_token_id = dflash_config["mask_token_id"]
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_main(f"Using mask_token_id: {mask_token_id}", main_rank)

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_main(f"dflash_config: {draft_model.config.dflash_config}", main_rank)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_main(f"Total training steps: {total_steps}", main_rank)

    print_on_main("Loading target embeddings and head...", main_rank)
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device.type,
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
        prefix_weight_base=args.prefix_weight_base,
    )

    # Wrap each transformer block as its own FSDP unit so that all-gather /
    # reduce-scatter overlap with compute. Without an auto_wrap_policy the
    # whole model is a single FSDP unit, forcing every collective onto the
    # critical path with no overlap. The block class is resolved from the
    # draft model's `_no_split_modules` so this stays architecture-agnostic
    # rather than hardcoding a specific decoder-layer class.
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    if dp_group is not None:
        fsdp_kwargs["process_group"] = dp_group
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dflash_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    dflash_model = FSDP(dflash_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_main(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}",
            main_rank,
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_main(f"Initializing tracker (report_to={args.report_to})...", main_rank)
    # Trackers gate on global rank 0 by default; in disaggregated mode that is
    # an inference rank which never logs, so nominate this group's main rank.
    args.tracker_main_rank = main_rank
    tracker = create_tracker(args, args.output_dir)
    print_on_main("Tracker initialized successfully.", main_rank)

    # Aligns with the inference ranks' pre-loop barrier in disaggregated mode
    # (and warms up the world communicator before the first p2p op); harmless
    # in colocated mode.
    dist.barrier()

    last_time = time.time()
    print_on_main(
        f"Starting training from epoch {start_epoch}, step {global_step}", main_rank
    )

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if rank == main_rank:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)
            hidden_states = hidden_provider(input_ids, attention_mask, loss_mask)

            loss, accuracy = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log, group=dp_group)
                dist.all_reduce(acc_log, group=dp_group)
                loss_log = loss_log / dist.get_world_size(dp_group)
                acc_log = acc_log / dist.get_world_size(dp_group)

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
                    main_rank=main_rank,
                )

            if rank == main_rank:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args,
                    epoch,
                    global_step,
                    dflash_model,
                    draft_model,
                    optimizer,
                    save_rank=main_rank,
                    process_group=dp_group,
                )

    save_checkpoint(
        args,
        args.num_epochs,
        global_step,
        dflash_model,
        draft_model,
        optimizer,
        save_rank=main_rank,
        process_group=dp_group,
    )

    tracker.close()
    # Let the other side (inference ranks in disaggregated mode) tear down in
    # step with us; harmless in colocated mode.
    dist.barrier()


def run_disagg_inference(args, ckpt_info: Tuple[int, int], num_shards: int):
    """Producer side of a disaggregated run (ranks [0, N)).

    Builds the sglang target (TP over the inference ranks), iterates every
    training shard's dataloader in lockstep, and streams each shard's hidden
    states to its training rank. Iteration order and skip logic must mirror
    run_training exactly — both sides derive them from the same dataset,
    samplers, and ckpt_info.
    """
    device = get_local_device()
    rank = dist.get_rank()
    num_inference_ranks = args.disagg_inference_ranks

    draft_config = build_draft_config(args)
    target_layer_ids = resolve_target_layer_ids(draft_config)

    print_on_rank0(
        f"Loading target model from {args.target_model_path} using sglang backend "
        f"(disaggregated inference, TP={num_inference_ranks})"
    )
    target_model = build_target_model(args)
    target_model.set_capture_layers(target_layer_ids)

    tokenizer = load_tokenizer(args.target_model_path)
    train_dataset = build_train_dataset(args, tokenizer)
    shard_loaders = [
        prepare_dp_dataloaders(
            train_dataset,
            args.batch_size,
            num_workers=max(1, args.dataloader_num_workers // num_shards),
            shuffle=True,
            dp_rank=shard_idx,
            dp_size=num_shards,
        )
        for shard_idx in range(num_shards)
    ]

    # DistributedSampler pads shards to equal length, so every loader has the
    # same number of batches — the same value run_training sees.
    num_batches = len(shard_loaders[0])
    start_epoch, global_step = ckpt_info
    skip_steps = global_step - start_epoch * num_batches

    # Matches the pre-loop barrier in run_training.
    dist.barrier()
    print_with_rank("Starting disaggregated inference loop")

    for epoch in range(start_epoch, args.num_epochs):
        for loader in shard_loaders:
            loader.sampler.set_epoch(epoch)
        iterators = [iter(loader) for loader in shard_loaders]

        for step_in_epoch in range(num_batches):
            # Consume every shard's batch even on skipped (resume) steps to
            # keep the data order aligned with the training side.
            batches = [next(it) for it in iterators]
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue

            for shard_idx, data in enumerate(batches):
                input_ids = data["input_ids"].to(device, non_blocking=True)
                attention_mask = data["attention_mask"].to(device, non_blocking=True)
                loss_mask = data["loss_mask"].to(device, non_blocking=True)
                target_output = target_model.generate_dflash_data(
                    input_ids, attention_mask, loss_mask
                )
                # Hidden states are replicated across the TP group, so spread
                # the send work round-robin over the inference ranks.
                if rank == shard_idx % num_inference_ranks:
                    send_hidden_states(
                        target_output.hidden_states.to(torch.bfloat16),
                        dst=num_inference_ranks + shard_idx,
                    )

    # Matches run_training's post-save barrier so teardown is in step.
    dist.barrier()


def run_disagg_training(
    args,
    draft_model_last_checkpoint: Optional[str],
    ckpt_info: Tuple[int, int],
    train_group: dist.ProcessGroup,
    num_shards: int,
):
    """Consumer side of a disaggregated run (ranks [N, world))."""
    device = get_local_device()
    rank = dist.get_rank()
    num_inference_ranks = args.disagg_inference_ranks
    shard_idx = rank - num_inference_ranks
    src_rank = shard_idx % num_inference_ranks

    draft_config = build_draft_config(args)

    # The inference ranks are constructing SGLangRunner right now, which
    # creates process groups via world-collective new_group() calls; join the
    # same sequence here or both sides deadlock.
    target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    join_sglang_collective_init(
        args.target_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )
    print_with_rank("Joined sglang collective init (training rank)")

    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    hidden_dim = len(draft_model.target_layer_ids) * draft_model.config.hidden_size
    print_on_main(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}",
        num_inference_ranks,
    )
    print_on_main(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}",
        num_inference_ranks,
    )

    tokenizer = load_tokenizer(args.target_model_path)
    train_dataset = build_train_dataset(args, tokenizer)
    train_dataloader = prepare_dp_dataloaders(
        train_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        dp_rank=shard_idx,
        dp_size=num_shards,
    )
    if args.eval_data_path:
        print_on_main(
            "Warning: --eval-data-path is ignored in disaggregated mode",
            num_inference_ranks,
        )

    def hidden_provider(input_ids, attention_mask, loss_mask):
        expected = (input_ids.shape[0], input_ids.shape[1], hidden_dim)
        return recv_hidden_states(expected, device, src=src_rank)

    run_training(
        args,
        draft_model,
        tokenizer,
        train_dataloader,
        hidden_provider,
        draft_model_last_checkpoint=draft_model_last_checkpoint,
        ckpt_info=ckpt_info,
        dp_group=train_group,
        main_rank=num_inference_ranks,
    )


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    num_inference_ranks = args.disagg_inference_ranks
    if num_inference_ranks > 0:
        if args.target_model_backend != "sglang":
            raise ValueError(
                "--disagg-inference-ranks requires --target-model-backend sglang"
            )
        if args.tp_size not in (1, num_inference_ranks):
            print(
                f"Warning: --tp-size {args.tp_size} is ignored in disaggregated "
                f"mode; the target runs TP={num_inference_ranks}"
            )
        args.tp_size = num_inference_ranks
        if args.sglang_mem_fraction_static < 0.6:
            # The 0.4 default in SGLangBackendArgs is a colocated setting that
            # leaves room for the draft/FSDP on the same GPU. Inference ranks
            # are dedicated to sglang; a low fraction leaves almost nothing
            # for the KV/mamba pools after the target weights (on hybrid
            # models this surfaces as "alloc_req_slots runs out of memory").
            print(
                f"Warning: --sglang-mem-fraction-static "
                f"{args.sglang_mem_fraction_static} is low for disaggregated "
                f"mode, where inference GPUs run only sglang; consider 0.75+"
            )

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    train_group = None
    num_shards = None
    if num_inference_ranks > 0:
        world_size = dist.get_world_size()
        if world_size <= num_inference_ranks:
            raise ValueError(
                f"--disagg-inference-ranks {num_inference_ranks} leaves no "
                f"training ranks (world size {world_size})"
            )
        if world_size % num_inference_ranks != 0:
            # The sglang patch builds its group lists as consecutive blocks of
            # tp_size covering the whole world; a non-divisible world breaks
            # the collective group creation the training ranks must mirror.
            raise ValueError(
                f"world size ({world_size}) must be divisible by "
                f"--disagg-inference-ranks ({num_inference_ranks})"
            )
        num_shards = world_size - num_inference_ranks
        # Collective: every rank must participate in group creation.
        train_group = dist.new_group(list(range(num_inference_ranks, world_size)))

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # If resuming, load config from checkpoint to ensure consistency
    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path

    if num_inference_ranks > 0:
        if dist.get_rank() < num_inference_ranks:
            run_disagg_inference(args, ckpt_info, num_shards)
        else:
            run_disagg_training(
                args,
                draft_model_last_checkpoint,
                ckpt_info,
                train_group,
                num_shards,
            )
        destroy_distributed()
        return

    target_model, draft_model = build_models(args)
    tokenizer = load_tokenizer(args.target_model_path)
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    device = get_local_device()

    def hidden_provider(input_ids, attention_mask, loss_mask):
        target_output = target_model.generate_dflash_data(
            input_ids, attention_mask, loss_mask
        )
        return target_output.hidden_states.to(device, non_blocking=True)

    run_training(
        args,
        draft_model,
        tokenizer,
        train_dataloader,
        hidden_provider,
        draft_model_last_checkpoint=draft_model_last_checkpoint,
        ckpt_info=ckpt_info,
    )
    destroy_distributed()


if __name__ == "__main__":
    main()
