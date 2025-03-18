import random
import sys
from typing import List

import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from torch.utils.data import DataLoader, Sampler


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

class DistributedResumableRandomSampler(Sampler):
    """A distributed sampler that supports training resumption and deterministic shuffling.
    
    This sampler handles data distribution across multiple processes for distributed training
    while maintaining the ability to resume training from checkpoints. It ensures deterministic
    shuffling based on epochs and proper data partitioning across processes.
    
    Args:
        data_source: Dataset to sample from
        shuffle (bool): Whether to shuffle the indices (default: True)
    """
    
    def __init__(self, data_source, shuffle: bool = True):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.shuffle = shuffle

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.epoch = 0
        self._generate_indices()
        self._current_index = 0
        self.just_resumed = False

    @property
    def indices(self) -> List[int]:
        return self.global_indices[self.rank::self.world_size]

    def _generate_indices(self):
        """Generates and partitions indices for the current epoch.
        
        Creates a list of indices, optionally shuffles them using the epoch as seed
        for deterministic shuffling, and partitions them across distributed processes.
        """
        indices = list(range(self.num_samples))
        if self.shuffle:
            # Use epoch as a seed so that every epoch reshuffles deterministically
            random.seed(self.epoch)
            random.shuffle(indices)
        # Store the global indices
        self.global_indices = indices

    def set_epoch(self, epoch: int):
        """Updates the epoch number and regenerates indices.

        Args:
            epoch (int): The new epoch number
        """
        # Skip resetting if the sampler has just resumed from a checkpoint
        if self.just_resumed:
            self.just_resumed = False
            return None

        # Set the epoch and indices
        self.epoch = epoch
        self._current_index = 0
        self._generate_indices()

    def __iter__(self):
        while self._current_index < len(self.indices):
            yield self.indices[self._current_index]
            self._current_index += 1

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        """Creates a serializable state dictionary for checkpointing.
        
        Returns:
            dict: Current state including index position, indices list, and epoch number
        """
        return {"current_index": self._current_index, "indices": self.global_indices, "epoch": self.epoch}

    def load_state_dict(self, state):
        """Restores sampler state from a checkpoint.
        
        Args:
            state (dict): Previously saved state dictionary
        """
        self._current_index = state["current_index"] + 1
        self.global_indices = state["global_indices"]
        self.epoch = state.get("epoch", 0)
        self.just_resumed = True

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
            
            
