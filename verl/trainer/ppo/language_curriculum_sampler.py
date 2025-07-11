import torch
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import Sampler
from typing import Dict, List, Any, Optional


class LanguageCurriculumSampler(Sampler):
    """
    Language-aware curriculum sampler that maintains separate difficulty
    progression for each language while respecting predefined language-
    specific sampling ratios.
    
    For machine translation, this ensures:
    1. Balanced sampling between target languages 
       (e.g., 50% Spanish, 50% Portuguese)
    2. Independent curriculum learning progression per language
    3. Language-specific target reward thresholds
    """
    
    def __init__(
        self,
        data_source,
        batch_size: int,
        language_ratios: Dict[str, float],
        language_target_rewards: Dict[str, float],
        initial_difficulties: Optional[Dict[str, float]] = None,
        alpha: float = 1.0,
        eta: float = 0.1,
        d_min: float = 0.0,
        d_max: float = 5.0,
        mini_epoch_size: int = 50
    ):
        """
        Initialize language-aware curriculum sampler.
        
        Args:
            data_source: Dataset to sample from
            batch_size: Size of each batch
            language_ratios: Dict mapping language codes to sampling ratios
                           e.g., {'es': 0.5, 'pt': 0.5}
            language_target_rewards: Dict mapping language codes to target 
                                   reward thresholds 
                                   e.g., {'es': 0.65, 'pt': 0.45}
            initial_difficulties: Optional dict of initial difficulties per 
                                language
            alpha: Sensitivity parameter for difficulty updates
            eta: Learning rate for difficulty updates  
            d_min: Minimum difficulty level
            d_max: Maximum difficulty level
            mini_epoch_size: Number of batches per mini-epoch (allows frequent 
                           updates)
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.language_ratios = language_ratios
        self.language_target_rewards = language_target_rewards
        self.mini_epoch_size = mini_epoch_size
        
        # Validate language ratios sum to 1.0
        ratio_sum = sum(language_ratios.values())
        if not np.isclose(ratio_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Language ratios must sum to 1.0, got {ratio_sum}"
            )
        
        # Global curriculum parameters
        self.alpha = alpha
        self.eta = eta
        self.d_min = d_min
        self.d_max = d_max
        
        sample_count = len(data_source)
        print(f"[LANG-CURRICULUM] Initializing with {sample_count} samples...")
        print(f"[LANG-CURRICULUM] Language ratios: {language_ratios}")
        print(f"[LANG-CURRICULUM] Target rewards: {language_target_rewards}")
        
        # Extract difficulties and languages from dataset
        extraction_result = self._extract_difficulties_and_languages()
        self.difficulties, self.languages = extraction_result
        
        # Group indices by language
        self.language_indices = self._group_indices_by_language()
        
        # Initialize per-language target difficulties
        init_method = self._initialize_language_difficulties
        self.language_difficulties = init_method(initial_difficulties)
        
        # Pre-sort indices by difficulty within each language 
        # for efficient sampling
        self.language_sorted_indices = self._presort_language_indices()
        
        self._log_initialization()
    
    def _extract_difficulties_and_languages(self):
        """Extract difficulty and language (from extra_info.tgt) 
        for all samples."""
        difficulties = []
        languages = []
        
        print("[LANG-CURRICULUM] Extracting difficulties and languages...")
        
        for i in range(len(self.data_source)):
            try:
                item = self.data_source[i]
                
                # Extract difficulty
                difficulty = item.get('difficulty', 0.0)
                difficulties.append(difficulty)
                
                # Extract language from extra_info.tgt
                extra_info = item.get('extra_info', {})
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except (json.JSONDecodeError, ValueError):
                        extra_info = {}
                
                language = extra_info.get('tgt', 'unknown')
                languages.append(language)
                
            except Exception as e:
                warning_msg = (
                    f"[LANG-CURRICULUM] Warning: "
                    f"Error processing item {i}: {e}"
                )
                print(warning_msg)
                difficulties.append(0.0)
                languages.append('unknown')
            
            # Progress logging for large datasets
            if (i + 1) % 10000 == 0:
                processed_count = i + 1
                total_count = len(self.data_source)
                progress_msg = (
                    f"[LANG-CURRICULUM] Processed "
                    f"{processed_count}/{total_count} samples"
                )
                print(progress_msg)
        
        return np.array(difficulties), languages
    
    def _group_indices_by_language(self):
        """Group sample indices by language."""
        language_indices = defaultdict(list)
        
        for idx, lang in enumerate(self.languages):
            language_indices[lang].append(idx)
        
        # Convert to regular dict
        language_indices = dict(language_indices)
        
        # Check that all configured languages exist in dataset
        configured_langs = set(self.language_ratios.keys())
        available_langs = set(language_indices.keys())
        
        missing_langs = configured_langs - available_langs
        if missing_langs:
            error_msg = (
                f"Languages {missing_langs} configured but not found in dataset. "
                f"Available languages: {available_langs}"
            )
            raise ValueError(error_msg)
        
        # Only keep configured languages
        filtered_indices = {
            lang: indices for lang, indices in language_indices.items() 
            if lang in configured_langs
        }
        
        return filtered_indices
    
    def _initialize_language_difficulties(self, initial_difficulties):
        """Initialize target difficulties for each language."""
        language_difficulties = {}
        
        for lang in self.language_ratios.keys():
            if initial_difficulties and lang in initial_difficulties:
                difficulty = initial_difficulties[lang]
            else:
                # Default to mean difficulty for this language
                lang_indices = self.language_indices[lang]
                lang_difficulties = self.difficulties[lang_indices]
                difficulty = float(np.mean(lang_difficulties))
            
            language_difficulties[lang] = difficulty
        
        return language_difficulties
    
    def _presort_language_indices(self):
        """Pre-sort indices by difficulty within each language for efficient sampling."""
        language_sorted_indices = {}
        
        for lang, indices in self.language_indices.items():
            indices = np.array(indices)
            lang_difficulties = self.difficulties[indices]
            sorted_order = np.argsort(lang_difficulties)
            language_sorted_indices[lang] = indices[sorted_order]
        
        return language_sorted_indices
    
    def _log_initialization(self):
        """Log initialization details for debugging."""
        total_samples = len(self.data_source)
        init_msg = f"[LANG-CURRICULUM] Initialized with {total_samples} total samples"
        print(init_msg)
        
        for lang, indices in self.language_indices.items():
            difficulties = self.difficulties[indices]
            sample_count = len(indices)
            diff_min = difficulties.min()
            diff_max = difficulties.max()
            diff_mean = difficulties.mean()
            lang_msg = (
                f"[LANG-CURRICULUM] {lang}: {sample_count} samples, "
                f"difficulty range [{diff_min:.3f}, {diff_max:.3f}], "
                f"mean={diff_mean:.3f}"
            )
            print(lang_msg)
        
        target_msg = f"[LANG-CURRICULUM] Initial target difficulties: {self.language_difficulties}"
        print(target_msg)
    
    def __iter__(self):
        """Yield batches for one mini-epoch with efficient memory management."""
        # Use smaller mini-epoch size to reduce memory pressure
        batches_yielded = 0
        effective_mini_epoch_size = min(self.mini_epoch_size, 20)  # Cap at 20 batches
        
        # Pre-sample indices for this mini-epoch to avoid repeated calculations
        if hasattr(self, '_target_difficulty'):
            # Simple curriculum mode - pre-compute sorted indices once
            diffs = np.abs(self.difficulties - self.target_difficulty)
            sorted_indices = np.argsort(diffs)
            
            # Yield batches from pre-sorted indices
            batch_start = 0
            while batches_yielded < effective_mini_epoch_size and batch_start < len(sorted_indices):
                batch_end = min(batch_start + self.batch_size, len(sorted_indices))
                batch_indices = sorted_indices[batch_start:batch_end]
                
                if len(batch_indices) > 0:
                    yield [int(idx) for idx in batch_indices]
                    batches_yielded += 1
                    batch_start = batch_end
                else:
                    break
        else:
            # Language-aware mode - efficient batch generation
            while batches_yielded < effective_mini_epoch_size:
                language_allocations = self._allocate_batch_by_language()
                
                # Sample from each language according to its current difficulty target
                batch_indices = []
                
                for lang, n_samples in language_allocations.items():
                    if n_samples > 0:
                        lang_indices = self._sample_from_language(lang, n_samples)
                        batch_indices.extend(lang_indices)
                
                if len(batch_indices) > 0:
                    yield [int(idx) for idx in batch_indices]
                    batches_yielded += 1
                else:
                    # If no valid batch could be created, stop iteration
                    break
        
        # Optional: explicit memory cleanup hint
        import gc
        gc.collect()
    
    def _allocate_batch_by_language(self) -> Dict[str, int]:
        """Determine how many samples to take from each language."""
        allocations = {lang: 0 for lang in self.language_ratios.keys()}
        # First pass: Calculate base allocations using floor
        remaining_batch = self.batch_size
        for lang, ratio in self.language_ratios.items():
            base_allocation = int(self.batch_size * ratio)
            available = len(self.language_indices[lang])
            allocated = min(base_allocation, remaining_batch, available)
            allocations[lang] = allocated
            remaining_batch -= allocated
        
        # Second pass: Distribute remaining samples using fractional parts
        if remaining_batch > 0:
            
            # Calculate fractional parts for fair distribution
            fractional_parts = []
            for lang, ratio in self.language_ratios.items():
                exact_allocation = self.batch_size * ratio
                fractional_part = exact_allocation - int(exact_allocation)
                available = len(self.language_indices[lang])
                can_take_more = allocations[lang] < available
                fractional_parts.append((fractional_part, lang, can_take_more))
            
            # Sort by fractional part (descending) to give priority to largest fractions
            fractional_parts.sort(reverse=True, key=lambda x: x[0] if x[2] else -1)
            
            # Distribute remaining samples to languages with highest fractional parts
            for fractional_part, lang, can_take_more in fractional_parts:
                if remaining_batch <= 0:
                    break
                if can_take_more:
                    allocations[lang] += 1
                    remaining_batch -= 1
        
        return allocations
    
    def _sample_from_language(self, language: str, n_samples: int) -> List[int]:
        """Sample n_samples from a specific language based on its target difficulty."""
        if n_samples <= 0:
            #print(f"[DEBUG] {language}: n_samples={n_samples}, returning empty list")
            return []
            
        target_difficulty = self.language_difficulties[language]
        sorted_indices = self.language_sorted_indices[language]
        
        if len(sorted_indices) == 0:
            return []
        
        if n_samples >= len(sorted_indices):
            return sorted_indices.tolist()
        
        # Find samples closest to target difficulty for this language
        lang_difficulties = self.difficulties[sorted_indices]
        diffs = np.abs(lang_difficulties - target_difficulty)
        closest_indices = np.argsort(diffs)[:n_samples]
        
        # Add small random noise to avoid always selecting same samples
        noise_factor = 0.1
        if len(closest_indices) > n_samples:
            noise = np.random.normal(0, noise_factor, len(diffs))
            noisy_diffs = diffs + noise
            closest_indices = np.argsort(noisy_diffs)[:n_samples]
        
        selected_indices = sorted_indices[closest_indices]
        #print(f"[DEBUG] {language}: Selected {len(selected_indices)} indices")
        return selected_indices.tolist()
    
    def update_language_difficulties(self, batch_rewards: torch.Tensor, 
                                   batch_languages: List[str]):
        """Update target difficulties for each language based on performance."""
        # Group rewards by language
        lang_rewards = defaultdict(list)
        for reward, lang in zip(batch_rewards, batch_languages):
            if lang in self.language_ratios:  # Only update configured languages
                lang_rewards[lang].append(reward.item())
        
        # Update each language's target difficulty
        for lang, rewards in lang_rewards.items():
            if lang in self.language_difficulties and rewards:
                avg_reward = np.mean(rewards)
                old_target = self.language_difficulties[lang]
                
                # Use language-specific target reward but global learning parameters
                lang_target_reward = self.language_target_rewards[lang]
                new_target = old_target + self.eta * np.tanh(
                    self.alpha * (avg_reward - lang_target_reward)
                )
                new_target = np.clip(new_target, self.d_min, self.d_max)
                
                self.language_difficulties[lang] = new_target
                
                update_msg = (
                    f"[LANG-CURRICULUM] {lang}: reward={avg_reward:.3f} "
                    f"(target={lang_target_reward:.3f}), "
                    f"difficulty {old_target:.3f} -> {new_target:.3f}"
                )
                print(update_msg)
    
    def get_language_difficulties(self) -> Dict[str, float]:
        """Get current target difficulties for all languages."""
        return dict(self.language_difficulties)
    
    def update_target_difficulty(self, new_target: float):
        """Update target difficulty for simple curriculum mode."""
        self._target_difficulty = new_target
    
    @property
    def target_difficulty(self):
        """Return representative target difficulty for backward compatibility."""
        if hasattr(self, '_target_difficulty'):
            return self._target_difficulty
        else:
            # For language-aware mode, return mean of all language difficulties
            return float(np.mean(list(self.language_difficulties.values())))
    
    @target_difficulty.setter
    def target_difficulty(self, value):
        """Set target difficulty for simple curriculum mode."""
        self._target_difficulty = value
    
    def get_language_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about each language in the dataset."""
        stats = {}
        for lang, indices in self.language_indices.items():
            difficulties = self.difficulties[indices]
            stats[lang] = {
                'sample_count': len(indices),
                'difficulty_min': float(difficulties.min()),
                'difficulty_max': float(difficulties.max()),
                'difficulty_mean': float(difficulties.mean()),
                'difficulty_std': float(difficulties.std()),
                'current_target': self.language_difficulties[lang]
            }
        return stats
    
    def __len__(self):
        """Return number of batches per mini-epoch."""
        effective_mini_epoch_size = min(self.mini_epoch_size, 20)  # Cap at 20 batches
        return effective_mini_epoch_size 