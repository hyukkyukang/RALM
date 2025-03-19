import math
import sys
from typing import TypeVar

import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from src.dataset.utils import batch_step_to_position

_T_co = TypeVar("_T_co", covariant=True)

class MyProgressBar(TQDMProgressBar):
    """Custom progress bar that properly handles distributed training scenarios.
    Extends PyTorch Lightning's TQDMProgressBar to correctly display total batch counts
    when resuming distributed training.
    """

    def init_train_tqdm(self) -> Tqdm:
        """Initializes a custom training progress bar.
        
        Overrides the default implementation to ensure proper display of total batch count
        during distributed data parallel (DDP) training, especially when resuming from checkpoints.
        
        Returns:
            Tqdm: Configured progress bar instance with correct total batch count.
        """
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
            total=self.total_train_batches,
        )

class DistributedResumableRandomSampler(Sampler[_T_co]):
    """Distributed sampler with resumable state via saved global indices.

    This sampler partitions the dataset indices across distributed processes in the
    same way as the original DistributedSampler but saves the global index order so
    that training can be resumed from a checkpoint. In addition, the sampler supports
    gradient accumulation, so that the resumed index is adjusted to the start of an
    accumulation cycle.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=42, drop_last=True, gradient_accumulation_steps=1):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.position = 0  # Track position in dataset
        self.drop_last = drop_last
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.has_just_resumed = False
        self.global_indices = None

        # Get the number of replicas
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.num_replicas = num_replicas

        # Get the rank of the current process
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.rank = rank

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type

        self.set_epoch(0)

    @property
    def indices(self):
        if not hasattr(self, "_indices"):
            assert self.global_indices is not None, f"global_indices are not set"
            self._indices = self.global_indices[self.rank::self.num_replicas]
        return self._indices

    def set_epoch(self, epoch):
        """ Set epoch to maintain shuffling consistency """
        # If the sampler has just resumed, we don't need to shuffle the indices
        # We have already loaded the information from the checkpoint
        if self.has_just_resumed:
            self.has_just_resumed = False
            return None

        # Set the epoch
        self.epoch = epoch

        # Shuffle the indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            global_indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            global_indices = list(range(self.num_samples))

        # Distribute the dataset across workers
        self.global_indices = global_indices
        self.position = 0  # Reset position at new epoch

    def set_position_by_batch_step(self, batch_step: int, per_device_batch_size: int) -> None:
        """ Set position in dataset """
        # Convert to the actual position in the dataset
        self.position = batch_step_to_position(batch_step, per_device_batch_size)

    def __iter__(self):
        return iter(self.indices[self.position:])

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        """ Save sampler state including global step """
        return {
            "position": self.position,
            "epoch": self.epoch,
            "global_indices": self.global_indices
        }

    def load_state_dict(self, state):
        """ Restore sampler state """
        self.position = state["position"]
        self.epoch = state["epoch"]
        self.global_indices = state["global_indices"]
        self.has_just_resumed = True


class ResumableDataLoader(DataLoader):
    """Extended DataLoader that supports checkpointing and resumption of training.
    
    Adds state saving and loading capabilities to PyTorch's DataLoader, enabling
    training to resume from the exact batch where it was stopped.
    """
    
    def __len__(self):
        """Returns the number of batches in the dataloader.
        
        Properly handles length calculation when using batch samplers.
        """
        # If a batch_sampler is present, its length is used as the number of batches.
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        return super().__len__()

    def state_dict(self):
        """Creates a serializable state dictionary if the sampler supports it.
        
        Returns:
            dict: Current sampler state or empty dict if sampler doesn't support checkpointing
        """
        if hasattr(self.sampler, "state_dict"):
            return self.sampler.state_dict()
        return {}

    def load_state_dict(self, state):
        """Restores dataloader state from a checkpoint if the sampler supports it.
        
        Args:
            state (dict): Previously saved state dictionary
        """
        if hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(state)

