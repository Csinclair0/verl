import torch
import numpy as np
import logging
from torch.utils.data import Sampler, BatchSampler

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

class CurriculumSampler(Sampler):
    """
    A curriculum sampler that yields individual indices based on difficulty.
    Use this when you want the DataLoader to handle batching.
    """
    def __init__(self, data_source, target_difficulty):
        self.data_source = data_source
        self.target_difficulty = target_difficulty
        self.num_samples = len(data_source)
        
        logger.info("Initializing CurriculumSampler with "
                    f"{self.num_samples} samples")
        
        # Extract difficulty levels from dataset with optimized approach
        self.difficulties = self._extract_difficulties_efficiently()
        
        logger.info(
            "CurriculumSampler initialization complete. "
            f"Difficulty range: [{self.difficulties.min():.3f}, "
            f"{self.difficulties.max():.3f}]"
        )
        
    def _extract_difficulties_efficiently(self):
        """Extract difficulty values efficiently, avoiding full 
        __getitem__ calls if possible."""
        difficulties = []
        
        # Check if dataset has direct access to difficulty without 
        # triggering __getitem__
        if hasattr(self.data_source, 'dataframe'):
            # For RLHFDataset, try to extract from the underlying dataframe
            try:
                logger.info("Attempting to extract difficulties from "
                            "dataframe directly")
                
                # Check if difficulty exists in the dataframe
                has_column_names = hasattr(self.data_source.dataframe, 
                                           'column_names')
                has_extra_info = (has_column_names and 
                                  'extra_info' in 
                                  self.data_source.dataframe.column_names)
                
                if has_extra_info:
                    # Extract from extra_info column directly
                    desc = "Extracting difficulties from dataframe"
                    for i in tqdm(range(len(self.data_source)), desc=desc):
                        extra_info = self.data_source.dataframe[i].get(
                            'extra_info', {}
                        )
                        if isinstance(extra_info, str):
                            import json
                            try:
                                extra_info = json.loads(extra_info)
                            except (json.JSONDecodeError, ValueError):
                                extra_info = {}
                        difficulty = extra_info.get('difficulty', 0)
                        difficulties.append(difficulty)
                    return np.array(difficulties)
            except Exception as e:
                logger.warning("Failed to extract difficulties from "
                               f"dataframe directly: {e}")
        
        # Fallback: use __getitem__ but with progress tracking
        logger.info("Extracting difficulties using __getitem__ method "
                    "(this may take a while for large datasets)")
        
        # Use smaller batches to avoid memory issues and show progress
        batch_size = min(1000, len(self.data_source) // 100)
        
        for i in tqdm(range(0, len(self.data_source), batch_size), 
                      desc="Extracting difficulties"):
            batch_difficulties = []
            end_idx = min(i + batch_size, len(self.data_source))
            
            for j in range(i, end_idx):
                try:
                    item = self.data_source[j]
                    difficulty = item.get('difficulty', 0)
                    batch_difficulties.append(difficulty)
                except Exception as e:
                    logger.warning("Error extracting difficulty for "
                                   f"item {j}: {e}, using default 0")
                    batch_difficulties.append(0)
            
            difficulties.extend(batch_difficulties)
            
            # Log progress every 10 batches
            if (i // batch_size) % 10 == 0:
                current_processed = min(end_idx, len(self.data_source))
                total_items = len(self.data_source)
                logger.info(f"Processed {current_processed}/{total_items} "
                            "items")
        
        return np.array(difficulties)
        
    def __iter__(self):
        # Sort indices by how close they are to target difficulty
        diffs = np.abs(self.difficulties - self.target_difficulty)
        sorted_indices = np.argsort(diffs)
        
        # Yield indices one by one (DataLoader will batch them)
        for idx in sorted_indices:
            yield int(idx)
    
    def update_target_difficulty(self, new_target):
        """Update the target difficulty dynamically based on model 
        feedback."""
        logger.info("Updating target difficulty from "
                    f"{self.target_difficulty} to {new_target}")
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
        self.num_samples = len(data_source)
        
        logger.info("Initializing CurriculumBatchSampler with "
                    f"{self.num_samples} samples, batch_size={batch_size}")
        
        # Extract difficulty levels from dataset - reuse the efficient method
        temp_sampler = CurriculumSampler(data_source, target_difficulty)
        self.difficulties = temp_sampler.difficulties
        
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
        """Update the target difficulty dynamically based on model 
        feedback."""
        logger.info("Updating target difficulty from "
                    f"{self.target_difficulty} to {new_target}")
        self.target_difficulty = new_target

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (
                (len(self.data_source) + self.batch_size - 1) // 
                self.batch_size
            )