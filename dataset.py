from transformers import T5ForConditionalGeneration, AutoTokenizer, PreTrainedTokenizer
import torch
import json
import random
from functools import partial
from typing import Sequence, Dict


def convert_to_features(tokenizer: PreTrainedTokenizer, example_batch: Sequence[Dict[str, str]], max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenizes texts and creates an input in the format for T5 models"""
    def add_prefix_eos_to_example(example):
        return f"grammar: {example['source']}", example['corrected']
    source_batch = []
    corrected_batch = []
    for example in example_batch:
        source, corrected = add_prefix_eos_to_example(example)
        source_batch.append(source)
        corrected_batch.append(corrected)
    tokenizer_call = partial(tokenizer.batch_encode_plus, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
    input_encodings = tokenizer_call(source_batch)
    target_encodings = tokenizer_call(corrected_batch)
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],

        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask'],
    }

def create_dataset(tokenizer:PreTrainedTokenizer, dataset_path:str , batch_size: int, max_length: int):
    """Loaded a dataset and sample batches."""
    accumulator = []
    with open(dataset_path, "rt") as fls:
        for jsn_ln in fls:
            accumulator.append(json.loads(jsn_ln))
    num_batches = len(accumulator) // batch_size
    while True:
        random.shuffle(accumulator)
        for i in range(num_batches):
            start_index = i*batch_size
            end_index = min(len(accumulator), start_index+batch_size)
            yield convert_to_features(tokenizer, accumulator[start_index: end_index], max_length)    
    
