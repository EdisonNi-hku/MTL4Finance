import collections

from transformers import TrainingArguments, logging
from sklearn.metrics import mean_squared_error

from transformers.data.data_collator import default_data_collator, InputDataClass
from models.models import BertForSequenceClassificationAttn, \
    BertForSequenceClassificationAttnConfig, BertForTokenClassificationAttn, BertForTokenClassificationAttnConfig
from models.mtl_models import DataLoaderWithTaskname, MultitaskDataloader, MultitaskModel


from argparse import ArgumentParser
from models.mtl_trainer import MultitaskTrainer
from utils import *
from compute_metrics import compute_metrics_sequence_classification, compute_metrics_sequence_regression, \
    compute_metrics_token_classification_acc, compute_metrics_token_classification_f1

logger = logging.get_logger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument('--task1', type=float, default=1.0)
    parser.add_argument('--task2', type=float, default=1.0)
    parser.add_argument('--task3', type=float, default=1.0)
    parser.add_argument('--task4', type=float, default=1.0)
    parser.add_argument('--task5', type=float, default=1.0)
    parser.add_argument('--task6', type=float, default=1.0)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--save', type=str, default='edge_model')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--model', type=str, default="prosusfinbert")
    parser.add_argument('--load', type=str, default="no")
    parser.add_argument('--cont', type=str, default="no")
    parser.add_argument('--no_sep', default=False, action='store_true')
    parser.add_argument('--testset', default=False, action='store_true')
    parser.add_argument('--layer_attn', default=False, action='store_true')
    parser.add_argument('--pool', default=False, action='store_true')
    parser.add_argument('--task_list', type=str, default='asa,fpb,na,srl,nc,cau')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--eval_acc_steps', type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1),
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    set_seed(args)
    tasks = args.task_list.split(',')
    for task in tasks:
        assert(task in ['asa', 'fpb', 'na', 'srl', 'nc', 'cau'])
    if args.model == 'prosusfinbert':
        tokenizer_name = 'ProsusAI/finbert'
    else:
        tokenizer_name = args.model
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, local_files_only=True, cache_dir='cache')

    asa_train_dataset = prepare_aspect_sentiment_data(True, 'train', tokenizer)
    asa_val_dataset = prepare_aspect_sentiment_data(True, 'valid', tokenizer)
    asa_test_dataset = prepare_aspect_sentiment_data(True, 'test', tokenizer)

    fpb_train_dataset = prepare_FPB_data('train', tokenizer)
    fpb_val_dataset = prepare_FPB_data('valid', tokenizer)
    fpb_test_dataset = prepare_FPB_data('test', tokenizer)

    na_train_dataset = prepare_number_attachment_data(True, 'train', tokenizer)
    na_val_dataset = prepare_number_attachment_data(True, 'valid', tokenizer)
    na_test_dataset = prepare_number_attachment_data(True, 'test', tokenizer)

    srl_train_dataset, srl_train_tags = prepare_semantic_parsing_data('train', tokenizer)
    srl_val_dataset, srl_val_tags = prepare_semantic_parsing_data('valid', tokenizer)
    srl_test_dataset, srl_test_tags = prepare_semantic_parsing_data('test', tokenizer)

    nc_train_dataset = prepare_number_classification_data(True, 'train', tokenizer)
    nc_train_encodings = nc_train_dataset.encodings
    nc_train_labels = nc_train_dataset.labels
    nc_val_dataset = prepare_number_classification_data(True, 'valid', tokenizer)
    nc_test_dataset = prepare_number_classification_data(True, 'test', tokenizer)
    nc_test_encodings = nc_test_dataset.encodings
    nc_test_labels = nc_test_dataset.labels

    cau_train_dataset, cau_train_tags = prepare_fin_causal_data('train', tokenizer)
    cau_val_dataset, cau_val_tags = prepare_fin_causal_data('valid', tokenizer)
    cau_test_dataset, cau_test_tags = prepare_fin_causal_data('test', tokenizer)

    dataset_dict = {
        "srl": srl_train_dataset,
        "na": na_train_dataset,
        "fpb": fpb_train_dataset,
        "asa": asa_train_dataset,
        "nc": nc_train_dataset,
        "cau": cau_train_dataset,
    }
    dataset_dict = filter_dict(tasks, dataset_dict)

    val_dataset_dict = {
        "srl": srl_val_dataset,
        "na": na_val_dataset,
        "fpb": fpb_val_dataset,
        "asa": asa_val_dataset,
        "nc": nc_val_dataset,
        "cau": cau_val_dataset,
    }
    val_dataset_dict = filter_dict(tasks, val_dataset_dict)

    collator_dict = {
        "srl": default_data_collator,
        "na": default_data_collator,
        "fpb": default_data_collator,
        "asa": default_data_collator,
        "nc": default_data_collator,
        "cau": default_data_collator,
    }
    collator_dict = filter_dict(tasks, collator_dict)

    batch_size_dict = {
        "srl": 4 * args.times,
        "na": 8 * args.times,
        "fpb": 4 * args.times,
        "asa": 4 * args.times,
        "nc": 6 * args.times,
        "cau": 4 * args.times,
    }
    batch_size_dict = filter_dict(tasks, batch_size_dict)

    model_type_dict = {
        "asa": BertForSequenceClassificationAttn,
        "na": BertForSequenceClassificationAttn,
        "fpb": BertForSequenceClassificationAttn,
        "srl": BertForTokenClassificationAttn,
        "nc": BertForSequenceClassificationAttn,
        "cau": BertForTokenClassificationAttn,
    }
    model_type_dict = filter_dict(tasks, model_type_dict)

    model_config_dict = {
        "asa": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=1, layer_attn=args.layer_attn, pool=True),
        "na": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=2, layer_attn=args.layer_attn, pool=True),
        "fpb": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=3, layer_attn=args.layer_attn, pool=True),
        "srl": BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=len(srl_unique_tags), layer_attn=args.layer_attn, pool=True),
        "nc": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=len(number_labels), layer_attn=args.layer_attn, pool=True),
        "cau": BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=5, layer_attn=args.layer_attn, pool=True),
    }
    model_config_dict = filter_dict(tasks, model_config_dict)

    loss_proportion_dict = {
        "asa": args.task1,
        "na": args.task2,
        "fpb": args.task3,
        "srl": args.task4,
        "nc": args.task5,
        "cau": args.task6,
    }
    loss_proportion_dict = filter_dict(tasks, loss_proportion_dict)

    if args.load == 'no':
        multitask_model = MultitaskModel.create(
            model_name=args.model,
            model_type_dict=model_type_dict,
            model_config_dict=model_config_dict,
            loss_proportion_dict=loss_proportion_dict,
        )
        attn_params = []
        for name, param in multitask_model.named_parameters():
            if "layer_attn" in name:
                attn_params.append(name)

        best_scores = {k: 0 for k in tasks}
        if args.cont != 'no':
            best_scores = args.save + '/best_score.pkl'

        compute_metrics_dict = {
            "asa": compute_metrics_sequence_regression,
            "na": compute_metrics_sequence_classification,
            "fpb": compute_metrics_sequence_classification,
            "srl": compute_metrics_token_classification_f1_srl,
            "nc": compute_metrics_sequence_classification,
            "cau": compute_metrics_token_classification_f1_cau,
        }
        compute_metrics_dict = filter_dict(tasks, compute_metrics_dict)
        trainer = MultitaskTrainer(
            model=multitask_model,
            args=TrainingArguments(
                output_dir=args.save,
                overwrite_output_dir=True,
                warmup_steps=500,
                learning_rate=0.00005,
                weight_decay=0.01,
                num_train_epochs=args.epoch,
                logging_steps=200,
                save_steps=200,
                load_best_model_at_end=True,
                evaluation_strategy='steps',
                eval_accumulation_steps=args.eval_acc_steps,
                save_total_limit=2,
                metric_for_best_model='eval_average_metrics',
                greater_is_better=True,
            ),
            params=attn_params,
            compute_metrics_tasks=compute_metrics_dict,
            train_dataset=dataset_dict,
            eval_dataset=val_dataset_dict,
            collator_dict=collator_dict,
            batch_size_dict=batch_size_dict,
            best_scores=best_scores,
        )
        if args.cont == 'no':
            trainer.train()
        else:
            trainer.train(args.cont)
    else:
        state_dict = torch.load(args.load + '/pytorch_model.bin')
        state_dicts = {task: collections.OrderedDict() for task in tasks}

        for k, v in state_dict.items():
            for task in tasks:
                if k.startswith("taskmodels_dict." + task + "."):
                    new_key = k.replace("taskmodels_dict." + task + ".", "")
                    state_dicts[task][new_key] = v

        test_config_dict = {
            'asa': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=1, layer_attn=args.layer_attn, pool=True),
            'na': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=2, layer_attn=args.layer_attn, pool=True),
            'fpb': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=3, layer_attn=args.layer_attn, pool=True),
            'srl': BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=len(srl_unique_tags), layer_attn=args.layer_attn, pool=True),
            'nc': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=len(number_labels), layer_attn=args.layer_attn, pool=True),
            'cau': BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=5, layer_attn=args.layer_attn, pool=True),
        }

        test_model_dict = {
            'asa': BertForSequenceClassificationAttn.from_pretrained(args.model, config=test_config_dict['asa']),
            'na': BertForSequenceClassificationAttn.from_pretrained(args.model, config=test_config_dict['na']),
            'fpb': BertForSequenceClassificationAttn.from_pretrained(args.model, config=test_config_dict['fpb']),
            'srl': BertForTokenClassificationAttn.from_pretrained(args.model, config=test_config_dict['srl']),
            'nc': BertForSequenceClassificationAttn.from_pretrained(args.model, config=test_config_dict['nc']),
            'cau': BertForTokenClassificationAttn.from_pretrained(args.model, config=test_config_dict['cau']),
        }

        test_model_dict = filter_dict(tasks, test_model_dict)
        for task in tasks:
            test_model_dict[task].load_state_dict(state_dicts[task])

        # trainer = Trainer(model=finnum, args=TrainingArguments(output_dir='default', eval_accumulation_steps=10), compute_metrics=compute_metrics)
        if args.testset:
            asa_t_dataset = asa_test_dataset
            na_t_dataset = na_test_dataset
            fpb_t_dataset = fpb_test_dataset
            srl_t_dataset = srl_test_dataset
            t_tags = srl_test_tags
            nc_t_labels = nc_test_labels
            nc_t_encodings = nc_test_encodings
            cau_t_dataset = cau_test_dataset
            cau_t_tags = cau_test_tags
        else:
            asa_t_dataset = asa_train_dataset
            na_t_dataset = na_train_dataset
            fpb_t_dataset = fpb_train_dataset
            srl_t_dataset = srl_train_dataset
            t_tags = srl_train_tags
            nc_t_labels = nc_train_labels
            nc_t_encodings = nc_train_encodings
            cau_t_dataset = cau_train_dataset
            cau_t_tags = cau_train_tags

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

        def compute_metrics_fpb(pred):
            out = pred
            pred = out.predictions
            labels = out.label_ids
            if isinstance(pred, tuple):
                logits = pred[0]
            else:
                logits = pred
            preds = np.argmax(logits, axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            acc = accuracy_score(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }

        def compute_metrics_asa(pred):
            out = pred
            pred = out.predictions
            labels = out.label_ids
            if isinstance(pred, tuple):
                logits = pred[0]
            else:
                logits = pred
            mse = mean_squared_error(labels, logits, squared=False)
            return {
                'rooted_mean_squared_error': mse,
            }

        if 'asa' in tasks:
            trainer = Trainer(
                model=test_model_dict['asa'],
                args=TrainingArguments(
                    output_dir='dummy',
                    eval_accumulation_steps=10,
                )
            )
            pred = trainer.predict(asa_t_dataset)
            print(compute_metrics_asa(pred))
            del pred, test_model_dict['asa']

        if 'na' in tasks:
            test_model_dict['na'].cuda()
            with torch.no_grad():
                na_t_labels = na_t_dataset.labels
                na_t_encodings = na_t_dataset.encodings
                pred = []
                for i in range(len(na_t_labels)):
                    out = test_model_dict['na'](
                        torch.tensor([na_t_encodings['input_ids'][i]]).cuda(),
                        token_type_ids=torch.tensor([na_t_encodings['token_type_ids'][i]]).cuda(),
                        return_dict=True,
                        attention_mask=torch.tensor([na_t_encodings['attention_mask'][i]]).cuda(),
                    )
                    label = np.array(out['logits'][0].cpu())
                    label = np.argmax(label, axis=0)
                    pred.append(label)

                precision, recall, f1, _ = precision_recall_fscore_support(na_t_labels, pred, average='macro')
                print("NA Accuracy:", accuracy_score(na_t_labels, pred))
                print("NA Macro:", f1)
                del pred, test_model_dict['na']

        if 'fpb' in tasks:
            trainer = Trainer(
                model=test_model_dict['fpb'],
                args=TrainingArguments(
                    output_dir='dummy',
                    eval_accumulation_steps=10,
                )
            )
            pred = trainer.predict(fpb_t_dataset)
            print("fpb:", compute_metrics_fpb(pred))
            del pred, test_model_dict['fpb']

        if 'srl' in tasks:
            trainer = Trainer(
                model=test_model_dict['srl'],
                args=TrainingArguments(
                    output_dir='dummy',
                    eval_accumulation_steps=10,
                )
            )
            out = trainer.predict(srl_t_dataset)
            pred = out.predictions
            label_ids = out.label_ids
            pred = np.argmax(pred[0], axis=-1)
            ner_labels = []
            for i, l in enumerate(label_ids):
                pred_label = []
                for j, tok in enumerate(l):
                    if 0 <= tok <= len(srl_unique_tags):
                        pred_label.append(srl_id2tag[pred[i][j]])
                ner_labels.append(pred_label)

            from seqeval.metrics import f1_score, accuracy_score

            print("NER macro f1:", f1_score(t_tags, ner_labels, average='macro'))
            print("NER micro f1:", f1_score(t_tags, ner_labels, average='micro'))
            print("NER accuracy:", accuracy_score(t_tags, ner_labels))
            del pred, test_model_dict['srl']

        if 'nc' in tasks:
            test_model_dict['nc'].cuda()
            with torch.no_grad():
                pred = []
                for i in range(len(nc_t_labels)):
                    out = test_model_dict['nc'](
                        torch.tensor([nc_t_encodings['input_ids'][i]]).cuda(),
                        token_type_ids=torch.tensor([nc_t_encodings['token_type_ids'][i]]).cuda(),
                        return_dict=True,
                        attention_mask=torch.tensor([nc_t_encodings['attention_mask'][i]]).cuda(),
                    )
                    label = np.array(out['logits'][0].cpu())
                    label = np.argmax(label, axis=0)
                    pred.append(label)

                from sklearn.metrics import f1_score

                print("NC Micro:", f1_score(nc_t_labels, pred, average='micro'))
                print("NC Macro:", f1_score(nc_t_labels, pred, average='macro'))
                del pred, test_model_dict['nc']

        if 'cau' in tasks:
            trainer = Trainer(
                model=test_model_dict['cau'],
                args=TrainingArguments(
                    output_dir='dummy',
                    eval_accumulation_steps=10,
                )
            )
            out = trainer.predict(cau_t_dataset)
            pred = out.predictions
            label_ids = out.label_ids
            pred = np.argmax(pred[0], axis=-1)
            ner_labels = []
            for i, l in enumerate(label_ids):
                pred_label = []
                for j, tok in enumerate(l):
                    if 0 <= tok <= 5:
                        pred_label.append(cau_id2tag[pred[i][j]])
                ner_labels.append(pred_label)

            from seqeval.metrics import f1_score, accuracy_score

            print("CAU MACRO f1:", f1_score(cau_t_tags, ner_labels, average='macro'))
            print("CAU MICRO f1:", f1_score(cau_t_tags, ner_labels, average='micro'))
            print("CAU accuracy:", accuracy_score(cau_t_tags, ner_labels))


if __name__ == '__main__':
    main()
