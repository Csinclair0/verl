import re
import copy
import logging
from typing import Dict, List
import requests
import json
import ray
from tm_training.models.translation.generation.prompt_utils import (
    LANG_TABLE, DOMAIN_TABLE
)
from tm_training.pipelines.translation.scripts.utils import valid_params
from verl.utils.reward_score.token_alignment import (
    compute_token_level_reward_with_errors
)

logger = logging.getLogger(__name__)

INFER_URL = (
    "http://tk-metric.kubeflow-creator-services-translation.svc.cluster.local"
    "/v2/models/metric_x_ft/infer"
)


def has_case_distinction(text: str) -> bool:
    """
    Check if the text contains characters that have case distinctions.
    
    Args:
        text: Input text to check
        
    Returns:
        True if the text contains characters with case distinctions, 
        False otherwise
    """
    # Check if any character in the text has case variants
    for char in text:
        if char.lower() != char.upper():
            return True
    return False


def is_all_uppercase(text: str) -> bool:
    """
    Check if text is all uppercase, considering only characters that have case.
    
    Args:
        text: Input text to check
        
    Returns:
        True if all case-sensitive characters are uppercase, 
        False otherwise
    """
    if not text.strip():
        return False
        
    # Get only characters that have case distinctions
    case_sensitive_chars = [char for char in text if char.lower() != char.upper()]
    
    if not case_sensitive_chars:
        # No case-sensitive characters, so concept doesn't apply
        return False
        
    # Check if all case-sensitive characters are uppercase
    return all(char.isupper() for char in case_sensitive_chars)


def compute_case_sensitivity_penalty(input_text: str, response_text: str) -> float:
    """
    Compute penalty for case sensitivity mismatch.
    
    Penalizes -1 if input is all uppercase but response is not, for 
    languages that support case distinctions.
    
    Args:
        input_text: The source input text
        response_text: The model's response text
        
    Returns:
        Penalty value (0 or -1)
    """
    # Only apply penalty if both texts have case-sensitive characters
    if (not has_case_distinction(input_text) or 
            not has_case_distinction(response_text)):
        return 0.0
    
    # Check if input is all uppercase but response is not
    if (is_all_uppercase(input_text) and 
            not is_all_uppercase(response_text)):
        return -1.0
        
    return 0.0


@ray.remote
def get_response_remote(data):
    """Get response from the QE model service for a single request.
    Returns (scores, token_errors)
    scores: list of scores
    token_errors: list of token errors 
    IE ({'Hello': 'major', 'World': 'minor'})
    """
    num_tries = 0
    while num_tries < 5:
        try:
            r = requests.post(INFER_URL, json=data)
            scores = r.json()['outputs'][0]['data']
            token_errors = r.json()['outputs'][1]['data']
            token_errors = [json.loads(x) for x in token_errors]
            return (scores, token_errors)
        except Exception as e:
            print(f"Error during request: {e}, {data}")
            num_tries += 1
    length = data['inputs'][0]['shape'][1]
    return ([5 for x in range(length)], [{} for x in range(length)])


def reward_fn_params(qe_input):
    mt = qe_input['mt']
    input_string = qe_input['input']
    valid = valid_params(input_string, mt) 
    if valid:
        return 0 
    else:
        return -5 


def format_input(example, include_language, include_domain, include_context):
    """Format input for QE model."""
    is_qe = (example.get('target', "NA") == "NA")
    if include_language:
        src_lang = " " + LANG_TABLE[example['src']].lower()
        tgt_lang = " " + LANG_TABLE[example['tgt']].lower()
    else:
        src_lang, tgt_lang = '', ''

    context_tag_map = {
        'Name': 'universe_name',
        'Genre': 'genre',
        'Age Rating': 'age_recommendation',
        'Context': 'game_location'
    }
    if include_context:
        domain_prefix = f"<domain>{DOMAIN_TABLE[example['domain']]}</domain>"
        context = json.loads(example.get('context', "{}"))
        for context_k, context_v in context.items():
            if context_v is not None and context_v != 'NA':
                context_tag = context_tag_map[context_k]
                domain_prefix += f"<{context_tag}>{context_v}</{context_tag}>"
    else:
        domain_prefix = ""

    if is_qe:
        example["text"] = (
            domain_prefix +
            f"<input>{src_lang}: {example['input']}</input>" +
            f"<mt>{tgt_lang}: {example['mt']}</mt>"
        )
    else:
        example["text"] = (
            domain_prefix +
            f"<input>{src_lang}: {example['input']}</input>" +
            f"<mt>{tgt_lang}: {example['mt']}</mt>" +
            f"<ref>{tgt_lang}: {example['target']}</ref>"
        )

    return example['text']


def extract_translations(text):
    """Extract translation from text with formatting tags."""
    pattern = r'<translation>(.*?)</translation>'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) > 0:
        return matches[0]
    return "NA"


def compare_lengths(qe_input):
    """Compare lengths of input and translation."""
    mt = qe_input['mt']
    input_string = qe_input['input']
    if len(mt) / len(input_string) > 2 and len(input_string) > 5:
        return -5
    return 0


def score_qe_model(inputs, metric_name, include_context_in_metrics):
    """Score a batch of inputs using the QE model."""
    if metric_name == 'metric_x_ft':
        include_context = True
        include_language = True
        include_domain = True
    else:
        include_context = False
        include_language = False
        include_domain = False
    qe_inputs = [
        format_input(x, include_language, include_domain, include_context)
        for x in inputs
    ]
    data_inputs = []
    batch_size = 4

    for i in range(0, len(qe_inputs), batch_size):
        batch_qe_inputs = qe_inputs[i:i + batch_size]
        data_inputs.append({
            "model_name": 'metric_x_ft',
            "inputs": [{
                "name": "QE_INPUTS",
                "datatype": "BYTES",
                "shape": [1, len(batch_qe_inputs)],
                "data": [batch_qe_inputs]
            }]
        })

    # Use Ray for parallel processing
    scores = []
    token_errors = []
    if not ray.is_initialized():
        ray.init()
    
    futures = [get_response_remote.remote(data) for data in data_inputs]
    results = ray.get(futures)
    for result in results:
        if result:
            scores.extend(result[0])
            token_errors.extend(result[1])

    scaled_scores = [25 - x for x in scores]
    if len(scaled_scores) < len(inputs):
        logger.info("missing scores, setting all to zero")
        scaled_scores = [0] * len(inputs)

    length_scores = [compare_lengths(x) for x in inputs]
    final_scores = [x + y for x, y in zip(scaled_scores, length_scores)]

    param_scores = [reward_fn_params(x) for x in inputs]
    final_scores = [x + y for x, y in zip(final_scores, param_scores)]

    return final_scores, token_errors


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute score for a single example."""
    reasoning = extra_info['include_reasoning']
    include_context = extra_info['include_context_in_metrics']
    qe_input = copy.deepcopy(extra_info)
    if not include_context:
        for key in ['universe_name', 'context']:
            del qe_input[key]

    if reasoning:
        qe_input['mt'] = extract_translations(solution_str)
    else:
        qe_input['mt'] = solution_str
    score, token_errors = score_qe_model(
        [qe_input], 'metric_x_ft', include_context
    )
    return score[0], token_errors[0] if token_errors else {}


def compute_batch_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict],
) -> List[float]:
    """Compute QE scores for a batch of examples."""
    if not (len(data_sources) == len(solution_strs) == 
            len(ground_truths) == len(extra_infos)):
        raise ValueError(
            f"Batch sizes mismatch: data_sources({len(data_sources)}), "
            f"solution_strs({len(solution_strs)}), "
            f"ground_truths({len(ground_truths)}), "
            f"extra_infos({len(extra_infos)})"
        )

    first_extra_info = extra_infos[0]
    include_reasoning_globally = first_extra_info.get(
        'include_reasoning', False
    )
    include_context_in_metrics_globally = first_extra_info.get(
        'include_context_in_metrics', False
    )

    batch_qe_inputs = []
    for i in range(len(solution_strs)):
        extra_info = extra_infos[i]
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i]
        qe_input = copy.deepcopy(extra_info)
        qe_input['target'] = ground_truth

        if include_reasoning_globally:
            extracted_mt = extract_translations(solution_str)
            qe_input['mt'] = extracted_mt
        else:
            qe_input['mt'] = solution_str
        batch_qe_inputs.append(qe_input)
    
    scores, token_errors = score_qe_model(
        inputs=batch_qe_inputs,
        metric_name='metric_x_ft',
        include_context_in_metrics=include_context_in_metrics_globally
    )
    
    return scores, token_errors


def compute_batch_score_with_token_errors(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict],
    data=None,
    tokenizer=None,
    error_weights: Dict[str, float] = None,
    use_token_level_errors: bool = True,
    apply_case_penalty: bool = True,
):
    """
    Compute QE scores for a batch with optional token-level error 
    distribution and case sensitivity penalty.
    
    Args:
        data_sources: List of data sources
        solution_strs: List of solution strings
        ground_truths: List of ground truth strings
        extra_infos: List of extra info dicts
        data: DataProto object containing batch data (for token-level 
            processing)
        tokenizer: Tokenizer for word-token alignment
        error_weights: Dict mapping error types to penalty weights
        use_token_level_errors: Whether to distribute errors to token level
        apply_case_penalty: Whether to apply case sensitivity penalty
        
    Returns:
        If use_token_level_errors=True and data/tokenizer provided:
            token-level reward tensor
        Else:
            (scores, token_errors) tuple
    """
    scores, token_errors = compute_batch_score(
        data_sources, solution_strs, ground_truths, extra_infos
    )
    
    # Apply case sensitivity penalty if requested
    if apply_case_penalty:
        case_penalties = []
        for i, extra_info in enumerate(extra_infos):
            input_text = extra_info.get('input', '')
            solution_str = solution_strs[i]
            
            # Extract actual translation if using reasoning format
            if extra_info.get('include_reasoning', False):
                actual_translation = extract_translations(solution_str)
            else:
                actual_translation = solution_str
                
            case_penalty = compute_case_sensitivity_penalty(
                input_text, actual_translation
            )
            case_penalties.append(case_penalty)
            
        # Add case penalties to scores
        scores = [score + penalty for score, penalty 
                  in zip(scores, case_penalties)]
    
    # If token-level processing is requested and we have the necessary data
    if use_token_level_errors and data is not None and tokenizer is not None:
        return compute_token_level_reward_with_errors(
            data=data,
            token_errors_list=token_errors,
            base_scores=scores,
            tokenizer=tokenizer,
            error_weights=error_weights,
            return_dict=False
        )
    else:
        return scores, token_errors