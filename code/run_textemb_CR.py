# coding=utf-8
""" Compute TextEmb for classification/regression tasks."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Subset)
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertModel, BertTokenizer)


from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from utils import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_textemb(args, train_dataset, model):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Compute TextEmb *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Batch size = %d", args.train_batch_size)

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    total_num_examples = 0
    global_feature_dict = {}
    for _ in train_iterator:
        num_examples = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                input_mask = inputs['attention_mask']
                outputs = model(**inputs)
                sequence_output = outputs[0]  # batch_size x max_seq_length x hidden_size
                # pooled_output = outputs[1] # batch_size x hidden_size

                active_sequence_output = torch.einsum("ijk,ij->ijk", [sequence_output, input_mask])
                avg_sequence_output = active_sequence_output.sum(1) / input_mask.sum(dim=1).view(input_mask.size(0), 1)

                if len(global_feature_dict) == 0:
                    global_feature_dict["avg_sequence_output"] = avg_sequence_output.sum(dim=0).detach().cpu().numpy()
                    # global_feature_dict["pooled_output"] = pooled_output.sum(dim=0).detach().cpu().numpy()
                else:
                    global_feature_dict["avg_sequence_output"] += avg_sequence_output.sum(dim=0).detach().cpu().numpy()
                    # global_feature_dict["pooled_output"] += pooled_output.sum(dim=0).detach().cpu().numpy()

            num_examples += input_mask.size(0)
        total_num_examples += num_examples

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples

    # Save features
    for key in global_feature_dict:
        np.save(os.path.join(args.output_dir, '{}.npy'.format(key)), global_feature_dict[key])


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def prepare_aspect_sentiment_raw_data(no_sep, split, tokenizer, predict_path=None):
    if split == 'pred':
        assert(predict_path is not None)
        path = predict_path
    else:
        path = 'data/SemEval-2017/' + split + '.json'
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    texts = []
    labels = []
    for t in raw:
        texts.append(t['title'])
        labels.append(float(t['sentiment']))

    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def prepare_number_attachment_raw_data(no_sep, split, tokenizer, predict_path=None):
    if split == 'pred':
        assert (predict_path is not None)
        path = predict_path
    else:
        path = 'data/FinNum-2/' + split + '.json'
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    texts = [t['tweet'] for t in raw]
    labels = [int(t['relation']) for t in raw]
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def prepare_number_classification_raw_data(no_sep, split, tokenizer, predict_path=None):
    def filter_nc_encoding(sentences, labels, tokenizer, max_len):
        new_sentences = []
        new_labels = []
        for i, s in enumerate(sentences):
            if len(tokenizer.encode(s)) <= max_len:
                new_sentences.append(sentences[i])
                new_labels.append(labels[i])
        return new_sentences, new_labels

    if split == 'pred':
        assert (predict_path is not None)
        path = predict_path
    else:
        path = 'data/FinNum_3/' + split + '.json'

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    texts = [t['paragraph'] for t in raw]
    labels = [number_tag2id[t['category']] for t in raw]

    texts, labels = filter_nc_encoding(texts, labels, tokenizer, 512)
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--model', type=str, default="prosusfinbert")
    parser.add_argument('--ASA_data_dir', type=str, default='data/SemEval-2017')
    parser.add_argument('--add_fiqa', default=False, action='store_true')
    parser.add_argument("--glue_task", default=False, type=eval,
                        help="Whether the task is a glue task.")
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument("--model_type", default='bert', type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--train_data_subset", type=int, default=-1,
                        help="If > 0: limit the training data to a subset of train_data_subset instances.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'run_args.txt'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=2))
        f.close()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    if args.glue_task:
        # Prepare GLUE task
        args.task_name = args.task_name.lower()
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=args.task_name,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                    do_lower_case=args.do_lower_case,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    else:
        if args.model == 'prosusfinbert':
            tokenizer_name = 'ProsusAI/finbert'
        else:
            tokenizer_name = args.model
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, local_files_only=True, cache_dir='cache')
        if args.task_name == 'asa':
            train_dataset = prepare_aspect_sentiment_raw_data(True, 'train', tokenizer)
            num_labels = 1
        elif args.task_name == 'fpb':
            train_dataset = prepare_FPB_data('train', tokenizer)
            num_labels = 3
        elif args.task_name == 'na':
            train_dataset = prepare_number_attachment_raw_data(True, 'train', tokenizer)
            num_labels = 2
        elif args.task_name == 'nc':
            train_dataset = prepare_number_classification_raw_data(True, 'train', tokenizer)
            num_labels = len(number_labels)
        else:
            raise ValueError("Task not implemented.")

        input_ids = torch.tensor(train_dataset.encodings['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(train_dataset.encodings['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(train_dataset.encodings['attention_mask'], dtype=torch.long)
        labels = torch.tensor(train_dataset.labels, dtype=torch.long)
        train_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=args.task_name)
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

    if args.train_data_subset > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.train_data_subset, len(train_dataset)))))
    compute_textemb(args, train_dataset, model)


if __name__ == "__main__":
    main()
