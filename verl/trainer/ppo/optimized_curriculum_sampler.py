import numpy as np
import logging
from typing import Any
from torch.utils.data import Sampler


logger = logging.getLogger(__name__)


class OptimizedCurriculumSampler(Sampler):
    """
    A curriculum sampler that selects batches based on difficulty.
    Yields batches (lists) of indices rather than individual indices.
    """
    
    def __init__(self, data_source: Any, target_difficulty: float, 
                 cache_dir: str = None, batch_size: int = 32):
        self.data_source = data_source
        self.target_difficulty = target_difficulty
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        
        logger.info(f"Initializing CurriculumSampler with "
                    f"{self.num_samples} samples, batch_size={batch_size}")
        
        # Extract difficulty levels from dataset
        logger.info("Extracting difficulties from dataset...")
        self.difficulties = self._extract_difficulties()
        
        logger.info(f"Curriculum sampler initialized. "
                    f"Difficulty range: [{self.difficulties.min():.3f}, "
                    f"{self.difficulties.max():.3f}], "
                    f"mean={self.difficulties.mean():.3f}")
        
        # Debug: Show difficulty distribution
        unique_values = np.unique(self.difficulties)
        logger.info(f"[DEBUG] Found {len(unique_values)} unique difficulty "
                    f"values")
        logger.info(f"[DEBUG] First 10 values: {unique_values[:10]}")
        
        if len(unique_values) == 1:
            logger.warning(f"[DEBUG] ALL SAMPLES HAVE SAME DIFFICULTY: "
                           f"{unique_values[0]} - Curriculum learning will "
                           f"not work!")
    
    def _extract_difficulties(self):
        """Extract difficulty values from the dataset."""
        difficulties = []
        
        logger.info("Extracting difficulties using direct access...")
        for i in range(len(self.data_source)):
            try:
                item = self.data_source[i]
                # Try to get difficulty from the item
                if isinstance(item, dict):
                    difficulty = item.get('difficulty', 0)
                else:
                    # If item has difficulty attribute
                    difficulty = getattr(item, 'difficulty', 0)
                difficulties.append(difficulty)
            except Exception as e:
                logger.warning(f"Error extracting difficulty for item {i}: {e}, "
                               f"using default 0")
                difficulties.append(0)
            
            # Log progress every 10000 items
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(self.data_source)} items")
        
        return np.array(difficulties)
    
    def __iter__(self):
        # Debug: Check if this method is being called
        logger.info(f"[DEBUG] CurriculumSampler.__iter__() CALLED! "
                    f"target_difficulty={self.target_difficulty}")
        
        # Sort indices by how close they are to target difficulty
        diffs = np.abs(self.difficulties - self.target_difficulty)
        sorted_indices = np.argsort(diffs)
        
        # Log curriculum sampling information
        logger.info(f"[CURRICULUM ITER] Starting iteration with "
                    f"target_difficulty={self.target_difficulty}")
        
        # Log statistics about how well the target matches available data
        within_range_05 = np.sum((self.difficulties >= self.target_difficulty - 0.5) & 
                                 (self.difficulties <= self.target_difficulty + 0.5))
        within_range_10 = np.sum((self.difficulties >= self.target_difficulty - 1.0) & 
                                 (self.difficulties <= self.target_difficulty + 1.0))
        
        logger.info(f"[CURRICULUM ITER] Samples within target±0.5: "
                    f"{within_range_05}/{len(self.difficulties)} "
                    f"({within_range_05/len(self.difficulties)*100:.1f}%)")
        logger.info(f"[CURRICULUM ITER] Samples within target±1.0: "
                    f"{within_range_10}/{len(self.difficulties)} "
                    f"({within_range_10/len(self.difficulties)*100:.1f}%)")
        
        # Show closest and furthest difficulties that will be selected
        closest_difficulties = self.difficulties[sorted_indices[:10]]
        furthest_difficulties = self.difficulties[sorted_indices[-10:]]
        logger.info(f"[CURRICULUM ITER] Closest 10 difficulties: "
                    f"{closest_difficulties}")
        logger.info(f"[CURRICULUM ITER] Furthest 10 difficulties: "
                    f"{furthest_difficulties}")
        
        # Yield batches of indices (like reference implementation)
        for i in range(0, len(sorted_indices), self.batch_size):
            batch_indices = sorted_indices[i:i + self.batch_size]
            yield [int(idx) for idx in batch_indices]
    
    def update_target_difficulty(self, new_target: float):
        """Update the target difficulty dynamically."""
        old_target = self.target_difficulty
        logger.info(f"[CURRICULUM UPDATE] Updating target difficulty from "
                    f"{old_target} to {new_target} "
                    f"(delta: {new_target - old_target:+.3f})")
        
        self.target_difficulty = new_target
        
        # Log how many samples are available near the new target
        within_range_05 = np.sum((self.difficulties >= new_target - 0.5) & 
                                 (self.difficulties <= new_target + 0.5))
        within_range_10 = np.sum((self.difficulties >= new_target - 1.0) & 
                                 (self.difficulties <= new_target + 1.0))
        
        logger.info(f"[CURRICULUM UPDATE] New target coverage: "
                    f"±0.5: {within_range_05}/{len(self.difficulties)} "
                    f"samples, ±1.0: {within_range_10}/"
                    f"{len(self.difficulties)} samples")
    
    def __len__(self):
        # Return number of batches, not number of samples
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_optimized_curriculum_sampler(data_source: Any, 
                                        target_difficulty: float = 0, 
                                        cache_dir: str = None):
    """
    Factory function to create a curriculum sampler.
    
    Args:
        data_source: The dataset to sample from
        target_difficulty: Initial target difficulty level  
        cache_dir: Directory to cache difficulty values (ignored in simple version)
        
    Returns:
        OptimizedCurriculumSampler instance
    """
    return OptimizedCurriculumSampler(
        data_source=data_source,
        target_difficulty=target_difficulty,
        cache_dir=cache_dir
    ) 