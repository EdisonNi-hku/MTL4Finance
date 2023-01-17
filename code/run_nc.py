import collections
import numpy as np
import torch
from argparse import ArgumentParser

from transformers import TrainingArguments, BertTokenizerFast, Trainer
from models.models import BertForSequenceClassificationAttn, BertForSequenceClassificationAttnPal, \
    BertForSequenceClassificationAttnPalConfig, BertForSequenceClassificationAttnConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils import set_seed, prepare_number_classification_data, number_labels, AttnTrainer


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

    train_dataset = prepare_number_classification_data(args.no_sep, 'train', tokenizer)
    train_encodings = train_dataset.encodings
    train_labels = train_dataset.labels
    val_dataset = prepare_number_classification_data(args.no_sep, 'valid', tokenizer)
    test_dataset = prepare_number_classification_data(args.no_sep, 'test', tokenizer)
    test_encodings = test_dataset.encodings
    test_labels = test_dataset.labels

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)
        from sklearn.metrics import f1_score
        return {'accuracy': f1_score(labels, predictions, average='micro')}

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
                model_config = BertForSequenceClassificationAttnPalConfig.from_pretrained(args.model, layer_attn=args.layer_attn,
                                                                                      soft=args.soft, pool=args.pool, num_labels=len(number_labels))
                model = BertForSequenceClassificationAttnPal.from_pretrained(args.model, config=model_config, cache_dir='cache')
                if args.load_path != '':
                    model.bert.load_state_dict(bert_state_dict)
                for i, _ in enumerate(model.bert.encoder.layer):
                    for param in model.bert.encoder.layer[i].attention.parameters():
                        param.requires_grad = False
                    for param in model.bert.encoder.layer[i].intermediate.parameters():
                        param.requires_grad = False
            else:
                model_config = BertForSequenceClassificationAttnConfig.from_pretrained(args.model, layer_attn=args.layer_attn, pool=args.pool, num_labels=len(number_labels))
                model = BertForSequenceClassificationAttn.from_pretrained(args.model, config=model_config, cache_dir='cache')
                if args.load_path != '':
                    model.bert.load_state_dict(bert_state_dict)

            training_args = TrainingArguments(
                output_dir=args.save,  # output directory
                num_train_epochs=args.epoch,  # total number of training epochs
                per_device_train_batch_size=6,  # batch size per device during training
                per_device_eval_batch_size=6,  # batch size for evaluation
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
                eval_accumulation_steps=2,
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
                model = BertForSequenceClassificationAttnPal.from_pretrained(args.cont)
            else:
                model = BertForSequenceClassificationAttn.from_pretrained(args.cont)
            training_args = TrainingArguments(
                output_dir=args.save,  # output directory
                num_train_epochs=args.epoch,  # total number of training epochs
                per_device_train_batch_size=6,  # batch size per device during training
                per_device_eval_batch_size=6,  # batch size for evaluation
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
                eval_accumulation_steps=2,
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
            model = BertForSequenceClassificationAttnPal.from_pretrained(args.test)
        else:
            model = BertForSequenceClassificationAttn.from_pretrained(args.test)

        model.cuda()
        with torch.no_grad():
            pred = []
            for i in range(len(test_labels)):
                out = model(
                    torch.tensor([test_encodings['input_ids'][i]]).cuda(),
                    token_type_ids=torch.tensor([test_encodings['token_type_ids'][i]]).cuda(),
                    return_dict=True,
                    attention_mask=torch.tensor([test_encodings['attention_mask'][i]]).cuda(),
                )
                label = np.array(out['logits'][0].cpu())
                label = np.argmax(label, axis=0)
                pred.append(label)

            from sklearn.metrics import f1_score

            print("NC Micro:", f1_score(test_labels, pred, average='micro'))
            print("NC Macro:", f1_score(test_labels, pred, average='macro'))
