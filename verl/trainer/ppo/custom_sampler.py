import torch
import numpy as np
from torch.utils.data import Sampler, BatchSampler

class CurriculumSampler(Sampler):
    """
    A curriculum sampler that yields individual indices based on difficulty.
    Use this when you want the DataLoader to handle batching.
    """
    def __init__(self, data_source, target_difficulty):
        self.data_source = data_source
        self.target_difficulty = target_difficulty
        # Extract difficulty levels from dataset
        self.difficulties = np.array([
            data_source[i]['difficulty'] for i in range(len(data_source))
        ])
        self.num_samples = len(data_source)
        
    def __iter__(self):
        # Sort indices by how close they are to target difficulty
        diffs = np.abs(self.difficulties - self.target_difficulty)
        sorted_indices = np.argsort(diffs)
        
        # Yield indices one by one (DataLoader will batch them)
        for idx in sorted_indices:
            yield int(idx)
    
    def update_target_difficulty(self, new_target):
        """Update the target difficulty dynamically based on model feedback."""
        self.target_difficulty = new_target

    def __len__(self):
        return self.num_samples


class CurriculumBatchSampler(BatchSampler):
    """
    A curriculum batch sampler that yields batches of indices based on 
    difficulty. Use this when you want to control batch composition directly.
    """
    def __init__(self, data_source, batch_size, target_difficulty, 
                 drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.target_difficulty = target_difficulty
        self.drop_last = drop_last
        # Extract difficulty levels from dataset
        self.difficulties = np.array([
            data_source[i]['difficulty'] for i in range(len(data_source))
        ])
        
    def __iter__(self):
        # Generate batches until we've covered the dataset
        indices_used = set()
        
        while len(indices_used) < len(self.data_source):
            # Find the batch that best matches the target difficulty
            selected_indices = self._select_closest_to_target(
                exclude=indices_used
            )
            
            if len(selected_indices) == 0:
                break
                
            # If drop_last=True and this batch is smaller than batch_size, 
            # skip it
            if self.drop_last and len(selected_indices) < self.batch_size:
                break
                
            indices_used.update(selected_indices)
            yield selected_indices
    
    def _select_closest_to_target(self, exclude=None):
        if exclude is None:
            exclude = set()
            
        # Compute the absolute difference from target difficulty for 
        # available samples
        available_indices = [
            i for i in range(len(self.data_source)) if i not in exclude
        ]
        
        if len(available_indices) == 0:
            return []
            
        available_difficulties = self.difficulties[available_indices]
        diffs = np.abs(available_difficulties - self.target_difficulty)
        
        # Get the indices sorted by how close they are to the target 
        # difficulty
        sorted_available = np.argsort(diffs)
        
        # Select the top-N closest samples (up to batch_size)
        selected_indices = []
        for i in sorted_available:
            actual_idx = available_indices[i]
            selected_indices.append(actual_idx)
            if len(selected_indices) >= self.batch_size:
                break
        
        return selected_indices
    
    def update_target_difficulty(self, new_target):
        """Update the target difficulty dynamically based on model feedback."""
        self.target_difficulty = new_target

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (
                (len(self.data_source) + self.batch_size - 1) // self.batch_size
            )