import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
import os
from datasets import Dataset

from scipy.stats import norm  # Used as a base distribution (to be quantized), you can use any other having a `cdf` method.
from scipy.stats import bernoulli, binom, rv_discrete
from distribution_generator.distributions import get_rv

import numpy as np
import os

from torch.utils.data import DataLoader, DistributedSampler

conda_env_path = os.environ.get('CONDA_PREFIX', '/default/path/if/not/found')
hf_cache_dir = os.path.join(conda_env_path, 'hf_cache')
os.environ['HF_HOME'] = hf_cache_dir
available_distributions = ["bernoulli", "binomial", "custom_joint", "custom_univariate","categorical", "xor"]

def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

def get_binomial_dataset(data_config):

    if data_config.mut_info is not None:
        raise NotImplementedError("Binomial distribution not implemented for mutual information")
    else:
        X = binom.rvs(n=data_config.params.n, p=data_config.params.p, size=data_config.n_samples)
        X = X.reshape(-1, 1)
    # Convert the NumPy array to a dictionary
    data_dict = {"feature": X}

    # Create the Hugging Face dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset

def get_custom_joint_dataset(data_config):
    joint_dist = np.array(data_config.params.dist)
    pmf = joint_dist.flatten()
    values = np.arange(len(pmf))
    rv = rv_discrete(name="hidden_univariate", values=(values, pmf))
    samples = rv.rvs(size=data_config.n_samples)
    samples = np.unravel_index(samples, joint_dist.shape)
    samples = np.stack(samples, axis=1)
    X = samples[:,0].reshape(-1,1)
    Y = samples[:,1].reshape(-1,1)
    # Y = np.random.permutation(Y).reshape(-1,1)
    
    count_array = np.zeros((np.max(samples)+1, np.max(samples)+1))
    for i in range(len(X)):
        count_array[X[i], Y[i]] += 1
    count_array = count_array / count_array.sum()

    # raise UserWarning(f"Count array is {count_array}, joint dist is {joint_dist}")
    
    # Convert the NumPy array to a dictionary
    data_dict = {"feature": np.concatenate([X,Y], axis=1)}

    # Create the Hugging Face dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset

def get_custom_univariate_dataset(data_config):
    joint_dist = np.array(data_config.params.dist)
    pmf = joint_dist.flatten()
    values = np.arange(len(pmf))
    rv = rv_discrete(name="hidden_univariate", values=(values, pmf))
    samples = rv.rvs(size=data_config.n_samples)
    samples = np.unravel_index(samples, joint_dist.shape)
    samples = np.stack(samples, axis=1)
    X = samples[:,0].reshape(-1,1)
    
    # Convert the NumPy array to a dictionary
    data_dict = {"feature": X}

    # Create the Hugging Face dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset

def get_distribution(data_config):
    if "binomial" in data_config.train:
        p = data_config.params.p
        n = data_config.params.n
        dist = binom.pmf(np.arange(n+1), n, p).reshape(1,-1,1)
        return dist
    if "categorical" in data_config.train:
        rv = get_rv(data_config.mut_info, data_config.alphabet_size, data_config.alphabet_size, data_config.seq_length_x, data_config.seq_length_y, min_val=data_config.min_val)
        return rv.joint_dist
    if "xor" in data_config.train:
        return None
    if hasattr(data_config, "mut_info") and data_config.mut_info is not None:
        rv = get_rv(data_config.mut_info,2,2, min_val=data_config.min_val)
        return rv.joint_dist.reshape(2,-1,1)
    if "custom_joint" in data_config.train:
        dist = np.array(data_config.params.dist)
        # raise UserWarning(f"Dist shape is {dist.shape}")
        return np.expand_dims(dist, axis=-1)
    if  "custom_univariate" in data_config.train:
        return np.array(data_config.params.dist).reshape(1,-1,1)
    else:
        print(data_config)
        p=data_config.params.p
        return bernoulli.pmf(np.arange(2), p).reshape(1,-1,1)

def get_bernoulli_dataset(data_config):

    # random_variable = UniformlyQuantized(mutual_information, norm(loc=0.0, scale=1.0))
    # random_variable = UniformlyQuantized(mutual_information, bernoulli(0.5))
    # random_variable = UniformlyQuantized(mutual_information, poisson(4.0))

    if data_config.mut_info is not None:
        rv = get_rv(data_config.mut_info,2,2,1,1,min_val=data_config.min_val)
        print("Joint distribution ", rv.joint_dist)
        print("Entropy: ", rv.entropy)
        print("MI: ", rv.mutual_information)
        X, Y = rv.rvs(size=data_config.n_samples)
        X = np.concatenate([X, Y], axis=1)
    else:
        X = bernoulli.rvs(p=data_config.params.p, size=data_config.n_samples)
        X = X.reshape(-1, 1)

    print("Data ", X)

    # Convert the NumPy array to a dictionary
    data_dict = {"feature": X}

    # Create the Hugging Face dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset

def get_categorical_dataset(data_config):

    rv = get_rv(data_config.mut_info, data_config.alphabet_size, data_config.alphabet_size, data_config.seq_length_x, data_config.seq_length_y, min_val=data_config.min_val)
    print("Joint distribution ", rv.joint_dist)
    print("Entropy: ", rv.entropy)
    print("MI: ", rv.mutual_information)
    X, Y = rv.rvs(size=data_config.n_samples)
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)

    data = np.concatenate([X, Y], axis=1)
    
    data_dict = {"feature": data}
    dataset = Dataset.from_dict(data_dict)
    return dataset

def get_xor_dataset(data_config):
    # Generate the initial array of shape (bs, n-1)
    X = bernoulli.rvs(p=data_config.params.p, size=(data_config.n_samples, data_config.params.n-1))

    Y = bernoulli.rvs(p=data_config.params.p, size=(data_config.n_samples, 1))
    X_xor = np.concatenate([X, Y], axis=1)
    X_xor = np.bitwise_xor.reduce(X_xor, axis=1, keepdims=True)

    # Append the XOR result as a new column to the original array
    X = np.concatenate([X, X_xor], axis=1)
    data = np.concatenate([X, Y], axis=1)
    data_dict = {"feature": data}
    dataset = Dataset.from_dict(data_dict)
    
    return dataset

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8, data_config=None, **kwargs):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif "bernoulli" in name:
        dataset = get_bernoulli_dataset(data_config)
    elif "binomial" in name:
        dataset = get_binomial_dataset(data_config)
    elif "custom_joint" in name:
        dataset = get_custom_joint_dataset(data_config)
    elif "custom_univariate" in name:
        dataset = get_custom_univariate_dataset(data_config)
    elif "categorical" in name:
        dataset = get_categorical_dataset(data_config)
    elif "xor" in name:
        dataset = get_xor_dataset(data_config)
    else:
        dataset = load_dataset(name, cache_dir=cache_dir, trust_remote_code=True)

    if name == "lambada":
        data = dataset
    else:
        for custom in available_distributions:
            if custom in name:
                return dataset.with_format('torch')
        else:
            data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None
    

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length, data_config=config.data)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length, data_config=config.data)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

