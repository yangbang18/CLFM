import json
import os
import transformers
from typing import List, Union, Dict, Any, Callable
from transformers import PreTrainedTokenizer, AutoTokenizer
from collections import defaultdict


class CLIPTokenizer(transformers.CLIPTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        use_original_merges: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        with open(merges_file, encoding="utf-8") as merges_handle:
            if use_original_merges:
                bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
            else:
                bpe_merges = merges_handle.read().strip().split("\n")[1:]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))


class CLIPTokenizerFast(transformers.CLIPTokenizerFast):
    slow_tokenizer_class = CLIPTokenizer


def batch_iterator(files: Union[str, List[str]], batch_size: int = 64):
    if not isinstance(files, list):
        files = [files]
    
    result = []
    for file in files:
        dataset = open(file, 'r', encoding='utf-8')
        for l in dataset:
            result.append(l)
            if len(result) == batch_size:
                yield result
                result = []


def train_new_tokenizer(
        tokenizer, 
        files: Union[str, List[str]], 
        vocab_size, 
        save_path: str,
        batch_size: int = 64, 
        **kwargs,
    ):
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(files, batch_size),
        vocab_size=vocab_size,
        **kwargs,
    )
    new_tokenizer.save_pretrained(save_path)


def add_language_special_tokens_to_tokenizer(
        tokenizer: PreTrainedTokenizer,
        language_special_token_format: str = '',
        languages: List[str] = [],
    ) -> int:
    tokens = []
    if language_special_token_format:
        from configs import language_info
        tokens = [language_special_token_format.format(lang=lang, full_name=language_info[lang]['name']) for lang in languages]
    num_added_tokens = tokenizer.add_tokens(tokens, special_tokens=True)
    return num_added_tokens


def combine_tokenizers(
        tokenizer: PreTrainedTokenizer, 
        new_tokenizer_path: str, 
        save_path: str,
        as_dict: bool = False,
        new_merges_first: bool = False,
        directly_use_new_tokenizer: bool = False,
        language_special_token_format: str = '',
        languages: List[str] = [],
    ) -> Union[PreTrainedTokenizer, Dict[str, Any]]:
    """
    Adapted from https://github.com/huggingface/tokenizers/issues/690
    """
    if directly_use_new_tokenizer:
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)
        # Add language special tokens
        num_added_tokens = add_language_special_tokens_to_tokenizer(new_tokenizer, language_special_token_format, languages)
        new_tokenizer.save_pretrained(save_path)
        # if num_added_tokens:
        #     # We remove added_tokens.json to avoid re-load bug
        #     os.remove(os.path.join(save_path, 'added_tokens.json'))
        return {"tokenizer": new_tokenizer} if as_dict else new_tokenizer

    vocab_file = os.path.join(save_path, 'vocab.json')
    merges_file = os.path.join(save_path, 'merges.txt')
    info_file = os.path.join(save_path, 'combine_tokenizers_info.json')

    # Add language special tokens
    num_added_tokens = add_language_special_tokens_to_tokenizer(tokenizer, language_special_token_format, languages)
    # We first save tokenizer's configuration
    tokenizer.save_pretrained(save_path)
    # We remove tokenizer.json by now and will save it later
    os.remove(os.path.join(save_path, 'tokenizer.json'))

    # Create a shared vocabulary
    vocab1 = tokenizer.vocab
    vocab2 = json.load(open(os.path.join(new_tokenizer_path, 'vocab.json')))    

    new_vocab = vocab1
    num_overlap = len(languages) - num_added_tokens
    
    token_mapping = defaultdict(list)
    for token in vocab2.keys():
        if token not in new_vocab.keys():
            token_id = len(new_vocab)
            new_vocab[token] = token_id
            token_mapping[token_id].extend(languages)
        else:
            token_id = new_vocab[token]
            token_mapping[token_id].extend(languages)
            num_overlap += 1
    
    if not languages:
        token_mapping = {}
    
    # Save the shared vocab, it will override the already saved tokenizer's vocab
    with open(vocab_file, 'w', encoding='utf-8') as wf:
        json.dump(new_vocab, wf, ensure_ascii=False, indent=4)
    
    # Merge two merges file
    merges1 = open(merges_file, 'r', encoding='utf-8').read().strip().split('\n')
    head, merges1 = merges1[:1], merges1[1:]
    merges2 = open(os.path.join(new_tokenizer_path, 'merges.txt'), 'r', encoding='utf-8').read().strip().split('\n')
    head, merges2 = merges2[:1], merges2[1:]

    if new_merges_first:
        # handle duplication
        merges_set = set(merges2)
        merges1 = [item for item in merges1 if item not in merges_set]
        with open(merges_file, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(head + merges2 + merges1))
    else:
        # handle duplication
        merges_set = set(merges1)
        merges2 = [item for item in merges2 if item not in merges_set]
        with open(merges_file, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(head + merges1 + merges2))

    # Instantiate the new tokenizer
    kwargs = {'use_original_merges': False} if isinstance(tokenizer, CLIPTokenizerFast) else {}
    tokenizer = tokenizer.from_pretrained(save_path, **kwargs)

    # Now, save_path will contain the file tokenizer.json
    tokenizer.save_pretrained(save_path)

    # save info
    num_unique = len(vocab2) - num_overlap
    info = dict(
        language_special_token_format=language_special_token_format,
        languages=languages,
        num_added_tokens=num_added_tokens,
        num_overlap=num_overlap,
        num_unique=num_unique,
        overlap_ratio=num_overlap * 100 / len(vocab2),
        unique_ratio=num_unique * 100 / len(vocab2),
        new_merges_first=new_merges_first,
    )
    with open(info_file, 'w', encoding='utf-8') as wf:
        json.dump(info, wf, ensure_ascii=False, indent=4)

    if as_dict:
        return {"tokenizer": tokenizer, "token_mapping": token_mapping, **info}

    return tokenizer


def update_processor(
        processor,
        new_tokenizer_path: str, 
        save_path: str,
        as_dict: bool = False,
        new_merges_first: bool = False,
        directly_use_new_tokenizer: bool = False,
        language_special_token_format: str = '',
        languages: List[str] = [],
    ):
    processor.save_pretrained(save_path)

    out = combine_tokenizers(
        tokenizer=processor.tokenizer,
        new_tokenizer_path=new_tokenizer_path,
        save_path=save_path,
        as_dict=as_dict,
        new_merges_first=new_merges_first,
        directly_use_new_tokenizer=directly_use_new_tokenizer,
        language_special_token_format=language_special_token_format,
        languages=languages,
    )
    if as_dict:
        processor.tokenizer = out.pop('tokenizer')
        out['processor'] = processor
        return out
    else:
        processor.tokenizer = out
        return processor
