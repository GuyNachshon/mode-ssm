"""
Distributed training utilities for MODE-SSM.
Handles PyTorch DDP setup with graceful single-GPU fallback.
"""

import os
import logging
import socket
from typing import Optional, Dict, Any, Tuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if distributed training is available and enabled"""
    return (
        torch.cuda.device_count() > 1 and
        'WORLD_SIZE' in os.environ and
        int(os.environ.get('WORLD_SIZE', 1)) > 1
    )


def get_rank() -> int:
    """Get current process rank"""
    return int(os.environ.get('RANK', 0))


def get_local_rank() -> int:
    """Get local rank within the node"""
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size() -> int:
    """Get total number of processes"""
    return int(os.environ.get('WORLD_SIZE', 1))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed(
    backend: str = 'nccl',
    timeout_minutes: int = 30,
    find_unused_parameters: bool = False
) -> Tuple[bool, Optional[int]]:
    """
    Initialize distributed training environment.

    Args:
        backend: Backend for distributed training ('nccl' or 'gloo')
        timeout_minutes: Timeout for distributed operations
        find_unused_parameters: Whether to find unused parameters in DDP

    Returns:
        (success, local_rank) tuple
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - falling back to single-GPU training")
        return False, None

    if not is_distributed():
        logger.info("Single GPU detected - using single-GPU training")
        return False, None

    try:
        # Get distributed parameters
        rank = get_rank()
        local_rank = get_local_rank()
        world_size = get_world_size()

        logger.info(f"Initializing distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")

        # Set CUDA device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Initialize process group
        if not dist.is_initialized():
            # Set master address and port if not set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = str(find_free_port())

            dist.init_process_group(
                backend=backend,
                init_method=f"env://",
                world_size=world_size,
                rank=rank,
                timeout=torch.distributed.default_pg_timeout * timeout_minutes
            )

        # Verify distributed setup
        if dist.get_rank() != rank:
            raise RuntimeError(f"Rank mismatch: expected {rank}, got {dist.get_rank()}")

        logger.info(f"Distributed training initialized successfully on rank {rank}")
        return True, local_rank

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        logger.info("Falling back to single-GPU training")
        cleanup_distributed()
        return False, None


def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_ids: List of device IDs (defaults to local_rank)
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Whether to use gradient as bucket view

    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        return model

    local_rank = get_local_rank()
    if device_ids is None:
        device_ids = [local_rank]

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=True,
        static_graph=False  # Important for dynamic models like Mamba
    )

    logger.info(f"Model wrapped with DDP on device {local_rank}")
    return ddp_model


def reduce_tensor(tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('mean', 'sum', 'max', 'min')

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor

    # Clone tensor to avoid in-place operations
    reduced_tensor = tensor.clone()

    if op == 'mean':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        reduced_tensor /= get_world_size()
    elif op == 'sum':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    elif op == 'max':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MAX)
    elif op == 'min':
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")

    return reduced_tensor


def gather_tensor(tensor: torch.Tensor) -> Optional[list]:
    """
    Gather tensor from all processes to rank 0.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes (only on rank 0)
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()
    rank = get_rank()

    # Prepare list for gathering
    gathered_tensors = None
    if rank == 0:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather tensors
    dist.gather(tensor, gathered_tensors, dst=0)

    return gathered_tensors if rank == 0 else None


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast Python object from source rank to all processes.

    Args:
        obj: Object to broadcast
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    # Use torch.distributed.broadcast_object_list
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def barrier():
    """Synchronize all processes"""
    if dist.is_initialized():
        dist.barrier()


class DistributedSampler:
    """Simple distributed sampler that works with our dataset"""

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        if dist.is_initialized():
            self.num_replicas = get_world_size()
            self.rank = get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.epoch = 0
        self.num_samples = len(dataset) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Truncate to make divisible by num_replicas
        indices = indices[:self.total_size]

        # Subsample for current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn=None,
    **kwargs
):
    """
    Create dataloader with distributed sampling.

    Args:
        dataset: Dataset to sample from
        batch_size: Batch size per process
        shuffle: Whether to shuffle data
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        collate_fn: Collate function
        **kwargs: Additional dataloader arguments

    Returns:
        DataLoader with distributed sampling
    """
    from torch.utils.data import DataLoader

    sampler = DistributedSampler(dataset, shuffle=shuffle) if is_distributed() else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        **kwargs
    )

    return dataloader, sampler


def log_distributed_info():
    """Log distributed training information"""
    if is_distributed():
        logger.info(f"Distributed training active:")
        logger.info(f"  World size: {get_world_size()}")
        logger.info(f"  Rank: {get_rank()}")
        logger.info(f"  Local rank: {get_local_rank()}")
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"  Current device: {torch.cuda.current_device()}")
    else:
        logger.info("Single-GPU training active")
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"  Current device: {torch.cuda.current_device()}")


def save_checkpoint_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    checkpoint_path: str,
    epoch: int,
    loss: float,
    additional_state: Optional[Dict[str, Any]] = None
):
    """
    Save checkpoint in distributed training (only on rank 0).

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        loss: Current loss
        additional_state: Additional state to save
    """
    if not is_main_process():
        return

    # Extract model state dict (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if additional_state is not None:
        checkpoint.update(additional_state)

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")