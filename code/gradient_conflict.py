import collections

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers.data.data_collator import default_data_collator
from models.models import BertForSequenceClassificationAttn, \
    BertForSequenceClassificationAttnConfig, BertForTokenClassificationAttn, BertForTokenClassificationAttnConfig
from models.mtl_models import MultitaskModel

from argparse import ArgumentParser

from utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument('--task1', type=float, default=1.0)
    parser.add_argument('--task2', type=float, default=1.0)
    parser.add_argument('--task3', type=float, default=1.0)
    parser.add_argument('--task4', type=float, default=1.0)
    parser.add_argument('--task5', type=float, default=1.0)
    parser.add_argument('--task6', type=float, default=1.0)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--model', type=str, default="prosusfinbert")
    parser.add_argument('--freeze', default=False, action='store_true')
    parser.add_argument('--layer_attn', default=False, action='store_true')
    parser.add_argument('--pool', default=False, action='store_true')
    parser.add_argument('--task_list', type=str, default='asa,fpb,na,srl,nc,cau')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--no_sep', default=False, action='store_true')
    args = parser.parse_args()

    set_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    tasks = args.task_list.split(',')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for task in tasks:
        assert (task in ['asa', 'fpb', 'na', 'srl', 'nc', 'cau'])

    if args.model == 'prosusfinbert':
        tokenizer_name = 'ProsusAI/finbert'
    else:
        tokenizer_name = args.model
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name, local_files_only=True, cache_dir='cache')
    asa_train_dataset = prepare_aspect_sentiment_data(args.no_sep, 'train', tokenizer)
    fpb_train_dataset = prepare_FPB_data('train', tokenizer)
    na_train_dataset = prepare_number_attachment_data(args.no_sep, 'train', tokenizer)
    srl_train_dataset, srl_train_tags = prepare_semantic_parsing_data('train', tokenizer)
    nc_train_dataset = prepare_number_classification_data(args.no_sep, 'train', tokenizer)
    cau_train_dataset, cau_train_tags = prepare_fin_causal_data('train', tokenizer)

    dataset_dict = {
        "srl": srl_train_dataset,
        "na": na_train_dataset,
        "fpb": fpb_train_dataset,
        "asa": asa_train_dataset,
        "nc": nc_train_dataset,
        "cau": cau_train_dataset,
    }
    dataset_dict = filter_dict(tasks, dataset_dict)

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

    dataloader_dict = {key: DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size_dict[key],
                                       collate_fn=collator_dict[key]) for key, dataset in dataset_dict.items()}

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
        "asa": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=1,
                                                                       layer_attn=args.layer_attn, pool=args.pool),
        "na": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=2,
                                                                      layer_attn=args.layer_attn, pool=args.pool),
        "fpb": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=3,
                                                                       layer_attn=args.layer_attn, pool=args.pool),
        "srl": BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=len(srl_unique_tags),
                                                                    layer_attn=args.layer_attn, pool=args.pool),
        "nc": BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=len(number_labels),
                                                                      layer_attn=args.layer_attn, pool=args.pool),
        "cau": BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=5,
                                                                    layer_attn=args.layer_attn, pool=args.pool),
    }
    model_type_dict = filter_dict(tasks, model_type_dict)

    loss_proportion_dict = {
        "asa": args.task1,
        "na": args.task2,
        "fpb": args.task3,
        "srl": args.task4,
        "nc": args.task5,
        "cau": args.task6,
    }
    loss_proportion_dict = filter_dict(tasks, loss_proportion_dict)

    multitask_model = MultitaskModel.create(
        model_name=args.model,
        model_type_dict=model_type_dict,
        model_config_dict=model_config_dict,
        loss_proportion_dict=loss_proportion_dict,
    )
    attn_params = []
    for name, param in multitask_model.named_parameters():
        if "layer_attn" in name or "sum_attn" in name:
            attn_params.append(name)

    state_dict = torch.load(args.load + '/pytorch_model.bin')
    state_dicts = {task: collections.OrderedDict() for task in tasks}

    for k, v in state_dict.items():
        for task in tasks:
            if k.startswith("taskmodels_dict." + task + "."):
                if "bert.pooler" in k and not args.pool:
                    continue
                new_key = k.replace("taskmodels_dict." + task + ".", "")
                state_dicts[task][new_key] = v

    for task in tasks:
        multitask_model.taskmodels_dict[task].load_state_dict(state_dicts[task])

    for task in tasks:
        task_model = multitask_model.taskmodels_dict[task].eval()
        num_example = 0
        global_gradient_dict = {}
        global_fisher_dict = {}
        for step, batch in tqdm(enumerate(dataloader_dict[task]), desc="Gradient of " + task, total=len(dataloader_dict[task])):
            inputs = batch
            outputs = task_model(**inputs)
            loss = outputs[0]
            task_model.zero_grad()
            loss.backward()
            gradient_dict = compute_gradients(task_model)
            if len(global_gradient_dict.keys()) == 0:
                for key in gradient_dict:
                    gradient = gradient_dict[key].detach().cpu().numpy()
                    global_gradient_dict[key] = gradient
                    global_fisher_dict[key] = gradient ** 2
            else:
                for key in gradient_dict:
                    gradient = gradient_dict[key].detach().cpu().numpy()
                    global_gradient_dict[key] += gradient
                    global_fisher_dict[key] += gradient ** 2
            task_model.zero_grad()
            num_example += inputs['input_ids'].size(0)
        # Normalization
        for key in global_gradient_dict:
            global_gradient_dict[key] = global_gradient_dict[key] / num_example
        for key in global_fisher_dict:
            global_fisher_dict[key] = global_fisher_dict[key] / num_example

        # Save features
        gradient_fn = os.path.join(args.output_dir, task + '.pkl')
        fisher_fn = os.path.join(args.output_dir, task + '_fisher.pkl')
        with open(gradient_fn, 'wb') as f:
            pickle.dump(global_gradient_dict, f)
        with open(fisher_fn, 'wb') as f:
            pickle.dump(global_fisher_dict, f)


if __name__ == '__main__':
    main()

