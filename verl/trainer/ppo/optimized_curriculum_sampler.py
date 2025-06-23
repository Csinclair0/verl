import numpy as np
import logging
import pickle
import os
from typing import Any

try:
    from torch.utils.data import Sampler
except ImportError:
    # Fallback for environments without torch
    class Sampler:
        def __init__(self, data_source):
            pass
        
        def __iter__(self):
            raise NotImplementedError
        
        def __len__(self):
            raise NotImplementedError

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


class OptimizedCurriculumSampler(Sampler):
    """
    An optimized curriculum sampler that caches difficulty values to disk
    to avoid re-processing large datasets on every initialization.
    """
    
    def __init__(self, data_source: Any, target_difficulty: float, 
                 cache_dir: str = None):
        self.data_source = data_source
        self.target_difficulty = target_difficulty
        self.num_samples = len(data_source)
        self.cache_dir = cache_dir or "/tmp/verl_curriculum_cache"
        
        logger.info("Initializing OptimizedCurriculumSampler with "
                    f"{self.num_samples} samples")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate cache key based on dataset properties
        cache_key = self._generate_cache_key()
        self.cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Load or compute difficulties
        self.difficulties = self._load_or_compute_difficulties()
        
        logger.info("OptimizedCurriculumSampler initialization complete. "
                    f"Difficulty range: [{self.difficulties.min():.3f}, "
                    f"{self.difficulties.max():.3f}]")
    
    def _generate_cache_key(self):
        """Generate a unique cache key for the dataset."""
        # Use dataset length and some sample data to create a hash
        key_data = f"{len(self.data_source)}"
        
        # Add some sample data for uniqueness if dataset is small enough
        if len(self.data_source) < 1000:
            try:
                # Sample first and last items to help create unique key
                first_item = self.data_source[0]
                last_item = self.data_source[-1]
                key_data += f"_{hash(str(first_item))}"
                key_data += f"_{hash(str(last_item))}"
            except Exception:
                pass
        
        # Add data file paths if available
        if hasattr(self.data_source, 'data_files'):
            key_data += f"_{hash(str(self.data_source.data_files))}"
        
        return abs(hash(key_data))
    
    def _load_or_compute_difficulties(self):
        """Load cached difficulties or compute if cache doesn't exist."""
        if os.path.exists(self.cache_file):
            logger.info("Loading cached difficulties from "
                        f"{self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    difficulties = pickle.load(f)
                
                # Verify cache is valid
                if len(difficulties) == len(self.data_source):
                    logger.info("Successfully loaded cached difficulties")
                    return difficulties
                else:
                    logger.warning("Cached difficulties length mismatch, "
                                   "recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load cached difficulties: {e}, "
                               "recomputing...")
        
        # Compute difficulties and cache them
        logger.info("Computing difficulties (this may take a while for "
                    "large datasets)...")
        difficulties = self._compute_difficulties_efficiently()
        
        # Cache the results
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(difficulties, f)
            logger.info(f"Cached difficulties to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache difficulties: {e}")
        
        return difficulties
    
    def _compute_difficulties_efficiently(self):
        """Compute difficulties using the most efficient method available."""
        difficulties = []
        
        # Method 1: Direct dataframe access (fastest)
        if hasattr(self.data_source, 'dataframe'):
            try:
                logger.info("Attempting fast dataframe difficulty extraction")
                
                # Check if we can access the dataframe directly
                df = self.data_source.dataframe
                has_column_names = hasattr(df, 'column_names')
                has_extra_info = (has_column_names and 
                                  'extra_info' in df.column_names)
                
                if has_extra_info:
                    logger.info("Extracting from dataframe column")
                    
                    # Process in chunks to manage memory
                    chunk_size = 10000
                    for i in tqdm(range(0, len(df), chunk_size), 
                                  desc="Processing dataframe chunks"):
                        end_idx = min(i + chunk_size, len(df))
                        chunk_difficulties = []
                        
                        for j in range(i, end_idx):
                            extra_info = df[j].get('extra_info', {})
                            if isinstance(extra_info, str):
                                import json
                                try:
                                    extra_info = json.loads(extra_info)
                                except (json.JSONDecodeError, ValueError):
                                    extra_info = {}
                            
                            difficulty = extra_info.get('difficulty', 0)
                            chunk_difficulties.append(difficulty)
                        
                        difficulties.extend(chunk_difficulties)
                    
                    return np.array(difficulties)
            except Exception as e:
                logger.warning(f"Dataframe method failed: {e}")
        
        # Method 2: Batch processing with __getitem__ (fallback)
        logger.info("Using fallback __getitem__ method with batching")
        
        # Process in smaller batches to show progress and manage memory
        batch_size = min(1000, max(1, len(self.data_source) // 100))
        
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
                    logger.warning(f"Error extracting difficulty for "
                                   f"item {j}: {e}, using default 0")
                    batch_difficulties.append(0)
            
            difficulties.extend(batch_difficulties)
            
            # Log progress every 50 batches or at the end
            batch_num = i // batch_size
            is_progress_point = (batch_num % 50 == 0 or 
                                 end_idx == len(self.data_source))
            if is_progress_point:
                pct = end_idx / len(self.data_source) * 100
                logger.info(f"Processed {end_idx}/{len(self.data_source)} "
                            f"items ({pct:.1f}%)")
        
        return np.array(difficulties)
    
    def __iter__(self):
        # Sort indices by how close they are to target difficulty
        diffs = np.abs(self.difficulties - self.target_difficulty)
        sorted_indices = np.argsort(diffs)
        
        # Log curriculum sampling information
        logger.info(f"[CURRICULUM ITER] Starting iteration with "
                    f"target_difficulty={self.target_difficulty}")
        
        # Log statistics about how well the target matches available data
        within_range_05 = ((self.difficulties >= self.target_difficulty - 0.5) & 
                           (self.difficulties <= self.target_difficulty + 0.5)).sum()
        within_range_10 = ((self.difficulties >= self.target_difficulty - 1.0) & 
                           (self.difficulties <= self.target_difficulty + 1.0)).sum()
        
        logger.info(f"[CURRICULUM ITER] Samples within target±0.5: "
                    f"{within_range_05}/{len(self.difficulties)} "
                    f"({within_range_05/len(self.difficulties)*100:.1f}%)")
        logger.info(f"[CURRICULUM ITER] Samples within target±1.0: "
                    f"{within_range_10}/{len(self.difficulties)} "
                    f"({within_range_10/len(self.difficulties)*100:.1f}%)")
        
        # Show closest and furthest difficulties that will be selected
        closest_difficulties = self.difficulties[sorted_indices[:10]]  # First 10
        furthest_difficulties = self.difficulties[sorted_indices[-10:]]  # Last 10
        logger.info(f"[CURRICULUM ITER] Closest 10 difficulties: "
                    f"{closest_difficulties}")
        logger.info(f"[CURRICULUM ITER] Furthest 10 difficulties: "
                    f"{furthest_difficulties}")
        
        # Yield indices one by one (DataLoader will batch them)
        for idx in sorted_indices:
            yield int(idx)
    
    def update_target_difficulty(self, new_target: float):
        """Update the target difficulty dynamically."""
        old_target = self.target_difficulty
        logger.info(f"[CURRICULUM UPDATE] Updating target difficulty from "
                    f"{old_target} to {new_target} (delta: {new_target - old_target:+.3f})")
        
        self.target_difficulty = new_target
        
        # Log how many samples are available near the new target
        if hasattr(self, 'difficulties'):
            within_range_05 = ((self.difficulties >= new_target - 0.5) & 
                               (self.difficulties <= new_target + 0.5)).sum()
            within_range_10 = ((self.difficulties >= new_target - 1.0) & 
                               (self.difficulties <= new_target + 1.0)).sum()
            
            logger.info(f"[CURRICULUM UPDATE] New target coverage: "
                        f"±0.5: {within_range_05}/{len(self.difficulties)} samples, "
                        f"±1.0: {within_range_10}/{len(self.difficulties)} samples")
    
    def clear_cache(self):
        """Clear the difficulty cache."""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("Difficulty cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def __len__(self):
        return self.num_samples


def create_optimized_curriculum_sampler(data_source: Any, 
                                        target_difficulty: float = 0, 
                                        cache_dir: str = None):
    """
    Factory function to create an optimized curriculum sampler.
    
    Args:
        data_source: The dataset to sample from
        target_difficulty: Initial target difficulty level  
        cache_dir: Directory to cache difficulty values
        
    Returns:
        OptimizedCurriculumSampler instance
    """
    return OptimizedCurriculumSampler(
        data_source=data_source,
        target_difficulty=target_difficulty,
        cache_dir=cache_dir
    ) 