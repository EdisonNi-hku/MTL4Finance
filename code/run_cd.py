import collections
from argparse import ArgumentParser

import torch
import numpy as np

from transformers import BertTokenizerFast, Trainer, TrainingArguments
from models.models import BertForTokenClassificationAttn, BertForTokenClassificationAttnConfig, \
    BertForTokenClassificationAttnPal, BertForTokenClassificationAttnPalConfig
from datasets import load_metric
from utils import set_seed, prepare_fin_causal_data, cause_tags, cau_id2tag, AttnTrainer

metric = load_metric("code/metrics/accuracy/accuracy.py")


def compute_metrics(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred, tuple):
        logits = pred[0]
    else:
        logits = pred
    predictions = np.argmax(logits, axis=-1)
    ner_labels = []
    tags = []
    for i, l in enumerate(labels):
        pred_label = []
        tag = []
        for j, tok in enumerate(l):
            if 0 <= tok <= len(cause_tags):
                pred_label.append(cau_id2tag[predictions[i][j]])
                tag.append(cau_id2tag[l[j]])
        ner_labels.append(pred_label)
        tags.append(tag)
    from seqeval.metrics import accuracy_score
    return {'accuracy': accuracy_score(tags, ner_labels)}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save', type=str, default='pool')
    parser.add_argument('--cont', type=str, default='no', help="whether continue training or not")
    parser.add_argument('--load_path', type=str, default='', help="load parameters from")
    parser.add_argument('--model', type=str, default='prosusfinbert')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--pal', default=False, action='store_true')
    parser.add_argument('--layer_attn', default=False, action='store_true')
    parser.add_argument('--soft', default=False, action='store_true')
    parser.add_argument('--pool', default=False, action='store_true')
    parser.add_argument('--no_sep', default=False, action='store_true')
    parser.add_argument('--test', type=str, default='no')
    parser.add_argument('--few', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', type=str, default='steps')
    parser.add_argument('--gas', type=int, default=1)

    args = parser.parse_args()

    set_seed(args)
    model_name = args.model
    if model_name == 'prosusfinbert':
        tokenizer_name = 'ProsusAI/finbert'
    else:
        tokenizer_name = model_name
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, local_files_only=True, cache_dir='cache')

    train_dataset, train_tags = prepare_fin_causal_data('train', tokenizer)
    val_dataset, val_tags = prepare_fin_causal_data('valid', tokenizer)
    test_dataset, test_tags = prepare_fin_causal_data('test', tokenizer)

    if args.test == 'no':
        if args.cont == 'no':
            if args.load_path != '':
                state_dict = torch.load(args.load_path + '/pytorch_model.bin')
                bert_state_dict = collections.OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("encoder."):
                        new_key = k.replace("encoder.", "", 1)
                        bert_state_dict[new_key] = v
            if args.pal:
                model_config = BertForTokenClassificationAttnPalConfig.from_pretrained(args.model,
                                                                                       layer_attn=args.layer_attn,
                                                                                       soft=args.soft, pool=args.pool,
                                                                                       num_labels=len(cause_tags))
                model = BertForTokenClassificationAttnPal.from_pretrained(model_name, config=model_config, cache_dir='cache')
                if args.load_path != '':
                    model.bert.load_state_dict(bert_state_dict)
                for i, _ in enumerate(model.bert.encoder.layer):
                    for param in model.bert.encoder.layer[i].attention.parameters():
                        param.requires_grad = False
                    for param in model.bert.encoder.layer[i].intermediate.parameters():
                        param.requires_grad = False
            else:
                model_config = BertForTokenClassificationAttnConfig.from_pretrained(args.model,
                                                                                       layer_attn=args.layer_attn,
                                                                                       pool=args.pool,
                                                                                       num_labels=len(cause_tags))
                model = BertForTokenClassificationAttn.from_pretrained(model_name, config=model_config, cache_dir='cache')
                if args.load_path != '':
                    model.bert.load_state_dict(bert_state_dict)

            training_args = TrainingArguments(
                output_dir=args.save,  # output directory
                num_train_epochs=args.epoch,  # total number of training epochs
                per_device_train_batch_size=8,  # batch size per device during training
                per_device_eval_batch_size=8,  # batch size for evaluation
                gradient_accumulation_steps=args.gas,
                warmup_steps=500,  # number of warmup steps for learning rate scheduler
                learning_rate=0.0001,
                weight_decay=0.01,  # strength of weight decay
                logging_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                evaluation_strategy=args.strategy,
                save_strategy=args.strategy,
                save_total_limit=2,
                eval_accumulation_steps=20,
                metric_for_best_model='accuracy',
                greater_is_better=True,
            )

            if args.layer_attn or args.soft:
                attn_params = []
                for name, param in model.named_parameters():
                    if "layer_attn" in name or "sum_attn" in name:
                        attn_params.append(name)
                trainer = AttnTrainer(
                    model=model,
                    params=attn_params,
                    compute_metrics=compute_metrics,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )

            trainer.train()
        else:
            if args.pal:
                model = BertForTokenClassificationAttnPal.from_pretrained(args.cont)
            else:
                model = BertForTokenClassificationAttn.from_pretrained(args.cont)
            training_args = TrainingArguments(
                output_dir=args.save,  # output directory
                num_train_epochs=args.epoch,  # total number of training epochs
                per_device_train_batch_size=8,  # batch size per device during training
                per_device_eval_batch_size=8,  # batch size for evaluation
                gradient_accumulation_steps=args.gas,
                warmup_steps=500,  # number of warmup steps for learning rate scheduler
                learning_rate=0.0001,
                weight_decay=0.01,  # strength of weight decay
                logging_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                evaluation_strategy=args.strategy,
                save_strategy=args.strategy,
                save_total_limit=2,
                eval_accumulation_steps=20,
                metric_for_best_model='accuracy',
                greater_is_better=True,
            )

            if args.layer_attn or args.soft:
                attn_params = []
                for name, param in model.named_parameters():
                    if "layer_attn" in name or "sum_attn" in name:
                        attn_params.append(name)
                trainer = AttnTrainer(
                    model=model,
                    params=attn_params,
                    compute_metrics=compute_metrics,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )
            else:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                )

            trainer.train(args.cont)
    else:
        if args.pal:
            model = BertForTokenClassificationAttnPal.from_pretrained(args.test)
        else:
            model = BertForTokenClassificationAttn.from_pretrained(args.test)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir='dummy',
                eval_accumulation_steps=20,
            )
        )
        out = trainer.predict(test_dataset)
        pred = out.predictions
        label_ids = out.label_ids
        pred = np.argmax(pred[0], axis=-1)
        ner_labels = []
        for i, l in enumerate(label_ids):
            pred_label = []
            for j, tok in enumerate(l):
                if 0 <= tok <= len(cause_tags):
                    pred_label.append(cau_id2tag[pred[i][j]])
            ner_labels.append(pred_label)

        from seqeval.metrics import f1_score, accuracy_score

        print("NER micro f1:", f1_score(test_tags, ner_labels, average='micro'))
        print("NER macro f1:", f1_score(test_tags, ner_labels, average='macro'))
        print("NER accuracy:", accuracy_score(test_tags, ner_labels))
