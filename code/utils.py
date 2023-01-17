import numpy as np
import random
import torch
import json
import pandas as pd
import pickle
import os
import collections

from models.models import BertForSequenceClassificationAttn, BertForSequenceClassificationAttnPal, \
    BertForSequenceClassificationAttnPalConfig, BertForSequenceClassificationAttnConfig, \
    BertForTokenClassificationAttnPalConfig, BertForTokenClassificationAttnPal, BertForTokenClassificationAttnConfig, \
    BertForTokenClassificationAttn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, Trainer, AdamW, Adafactor
from transformers.trainer import get_parameter_names
from nltk.tokenize import word_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"
srl_unique_tags = ['O', 'I-QUANT', 'B-QUANT', 'I-TIME', 'B-TIME', 'I-MANNER', 'B-MANNER', 'I-THEME', 'B-THEME',
                   'I-VALUE', 'B-VALUE', 'I-WHOLE', 'B-WHOLE',
                   'I-LOCATION', 'B-LOCATION', 'I-AGENT', 'B-AGENT', 'I-CAUSE', 'B-CAUSE', 'I-SOURCE', 'B-SOURCE',
                   'I-REF_TIME', 'B-REF_TIME', 'I-CONDITION', 'B-CONDITION']
srl_tag2id = {tag: id for id, tag in enumerate(srl_unique_tags)}
srl_id2tag = {id: tag for tag, id in srl_tag2id.items()}

cause_tags = ['O', 'I-CAUSE', 'B-CAUSE', 'I-EFFECT', 'B-EFFECT']
cau_tag2id = {tag: id for id, tag in enumerate(cause_tags)}
cau_id2tag = {id: tag for tag, id in cau_tag2id.items()}

number_labels = ['other', 'date', 'money', 'relative', 'quantity_absolute', 'absolute', 'product number', 'ranking', 'change', 'quantity_relative', 'time']
number_tag2id = {tag: id for id, tag in enumerate(number_labels)}
number_id2tag = {id: tag for id, tag in enumerate(number_labels)}

def filter_dict(keys, dic):
    new_dict = {}
    for k, v in dic.items():
        if k in keys:
            new_dict[k] = v
    return new_dict


class AttnTrainer(Trainer):

    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n in decay_parameters and n in self.params],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n in decay_parameters and n not in self.params],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n not in decay_parameters and n in self.params],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n not in decay_parameters and n not in self.params],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer


def random_shuffle(texts, labels):
    zipped = list(zip(texts, labels))
    random.shuffle(zipped)
    return [t[0] for t in zipped], [t[1] for t in zipped]


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load_num_attach_dataset(no_sep, path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    tweets = [t['tweet'] for t in raw]
    labels = [int(t['relation']) for t in raw]
    cashtags = [t['target_cashtag'] for t in raw]
    offset = [int(t['offset']) for t in raw]
    numbers = [str(t['target_num']) for t in raw]

    texts = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        if no_sep:
            number_length = len(numbers[i])
            tweet = tweet[:offset[i]] + '<' + numbers[i] + '>' + tweet[offset[i] + number_length:]
            texts.append(tweet.replace(cashtags[i], '|' + cashtags[i] + '|'))
        else:
            texts.append(tweet + '[SEP]' + cashtags[i] + '[SEP]' + numbers[i])
    return texts, labels


def convert_tags(lines):
    converted = []
    text = []
    qv_edges = []
    tv_edges = []
    mv_edges = []
    for l in lines:
        data_dict = json.loads(l)
        nodes = data_dict['nodes']
        tokens = data_dict['tokens']
        edges = data_dict['edges']
        tags = []
        for i, node in enumerate(nodes):
            if len(tags) < node[0][0]:
                tags += ['O'] * (node[0][0] - len(tags))
            if list(node[1].keys())[0] == 'quant' or list(node[1].keys())[0] == 'co_quant':
                tags += ['B-QUANT']
                tags += ['I-QUANT'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'time':
                tags += ['B-TIME']
                tags += ['I-TIME'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'manner':
                tags += ['B-MANNER']
                tags += ['I-MANNER'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'theme':
                tags += ['B-THEME']
                tags += ['I-THEME'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'value':
                tags += ['B-VALUE']
                tags += ['I-VALUE'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'whole':
                tags += ['B-WHOLE']
                tags += ['I-WHOLE'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'location':
                tags += ['B-LOCATION']
                tags += ['I-LOCATION'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'agent':
                tags += ['B-AGENT']
                tags += ['I-AGENT'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'cause':
                tags += ['B-CAUSE']
                tags += ['I-CAUSE'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'source':
                tags += ['B-SOURCE']
                tags += ['I-SOURCE'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'reference_time':
                tags += ['B-REF_TIME']
                tags += ['I-REF_TIME'] * (node[0][1] - node[0][0] - 1)
            elif list(node[1].keys())[0] == 'condition':
                tags += ['B-CONDITION']
                tags += ['I-CONDITION'] * (node[0][1] - node[0][0] - 1)
            else:
                #print(list(node[1].keys())[0])
                tags += ['O'] * (node[0][1] - node[0][0])
            if i == len(nodes) - 1:
                tags += ['O'] * (len(tokens) - len(tags))

        qv_edge = []
        tv_edge = []
        mv_edge = []
        value_idx = [i for i, x in enumerate(tags) if x == 'B-VALUE']
        quant_idx = [i for i, x in enumerate(tags) if x == 'B-QUANT']
        theme_idx = [i for i, x in enumerate(tags) if x == 'B-THEME']
        manner_idx = [i for i, x in enumerate(tags) if x == 'B-MANNER']
        for i, edge in enumerate(edges):
            if list(edge[2].keys())[0] == 'fact':
                if tags[edge[0][0]].endswith('VALUE') and tags[edge[1][0]].endswith('QUANT'):
                    qv_edge.append([quant_idx.index(edge[1][0]), value_idx.index(edge[0][0])])
                elif tags[edge[0][0]].endswith('QUANT') and tags[edge[1][0]].endswith('VALUE'):
                    qv_edge.append([quant_idx.index(edge[0][0]), value_idx.index(edge[1][0])])
                elif tags[edge[0][0]].endswith('VALUE') and tags[edge[1][0]].endswith('THEME'):
                    tv_edge.append([theme_idx.index(edge[1][0]), value_idx.index(edge[0][0])])
                elif tags[edge[0][0]].endswith('THEME') and tags[edge[1][0]].endswith('VALUE'):
                    tv_edge.append([theme_idx.index(edge[0][0]), value_idx.index(edge[1][0])])
                elif tags[edge[0][0]].endswith('VALUE') and tags[edge[1][0]].endswith('MANNER'):
                    mv_edge.append([manner_idx.index(edge[1][0]), value_idx.index(edge[0][0])])
                elif tags[edge[0][0]].endswith('MANNER') and tags[edge[1][0]].endswith('VALUE'):
                    mv_edge.append([manner_idx.index(edge[0][0]), value_idx.index(edge[1][0])])

        qv_edges.append(qv_edge)
        tv_edges.append(tv_edge)
        mv_edges.append(mv_edge)
        converted.append(tags)
        text.append(tokens)

    return text, converted, {'qv': qv_edges, 'tv': tv_edges, 'mv': mv_edges}


def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


class TransformersDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_aspect_sentiment_data(no_sep, split, tokenizer, predict_path=None):
    if split == 'pred':
        assert(predict_path is not None)
        path = predict_path
    else:
        path = 'data/TSA/' + split + '.json'
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    texts = []
    labels = []
    for t in raw:
        if no_sep:
            texts.append(t['title'].replace(t['company'], '|' + t['company'] + '|'))
        else:
            texts.append(t['title'] + '[SEP]' + t['company'])
        labels.append(float(t['sentiment']))

    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def prepare_number_attachment_data(no_sep, split, tokenizer, predict_path=None):
    if split == 'pred':
        assert (predict_path is not None)
        path = predict_path
    else:
        path = 'data/NAD/' + split + '.json'
    texts, labels = load_num_attach_dataset(no_sep, path)
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def prepare_semantic_parsing_data(split, tokenizer):
    path = 'data/FSRL/' + split + '.json'
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]

    texts, tags, edges_r = convert_tags(lines)

    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    labels = encode_tags(tags, encodings, srl_tag2id)
    encodings.pop("offset_mapping")

    dataset = TransformersDataset(encodings, labels)

    return dataset, tags


def prepare_FPB_data(split, tokenizer):
    path = 'data/SC/' + split + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    texts = [dic['text'] for dic in data]
    labels = [dic['label'] for dic in data]

    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


def prepare_fin_causal_data(split, tokenizer):
    # dataframe = pd.read_csv('data/fnp2020/fnp2020-fincausal2-task2.csv', sep=';')
    # sentences = [word_tokenize(s) for s in dataframe[' Sentence']]
    # tokens = []
    # tags = []
    # for i, s in enumerate(sentences):
    #     token = []
    #     tag = []
    #     j = 0
    #     cau_flag = False
    #     eff_flag = False
    #     while j < len(s):
    #         if j < len(s) - 2 and s[j] == '<' and s[j + 1] == 'e1' and s[j + 2] == '>':
    #             assert (not eff_flag)
    #             cau_flag = True
    #             j += 3
    #         elif j < len(s) - 2 and s[j] == '<' and s[j + 1] == '/e1' and s[j + 2] == '>':
    #             assert (not eff_flag)
    #             cau_flag = False
    #             j += 3
    #         elif j < len(s) - 2 and s[j] == '<' and s[j + 1] == 'e2' and s[j + 2] == '>':
    #             assert (not cau_flag)
    #             eff_flag = True
    #             j += 3
    #         elif j < len(s) - 2 and s[j] == '<' and s[j + 1] == '/e2' and s[j + 2] == '>':
    #             assert (not cau_flag)
    #             eff_flag = False
    #             j += 3
    #         else:
    #             token.append(s[j])
    #             if eff_flag and not cau_flag and (len(tag) == 0 or tag[-1] == 'O'):
    #                 tag.append('B-EFFECT')
    #             elif eff_flag and not cau_flag:
    #                 tag.append('I-EFFECT')
    #             elif cau_flag and not eff_flag and (len(tag) == 0 or tag[-1] == 'O'):
    #                 tag.append('B-CAUSE')
    #             elif cau_flag and not eff_flag:
    #                 tag.append('I-CAUSE')
    #             else:
    #                 tag.append('O')
    #             j += 1
    #     assert (len(token) == len(tag))
    #     tokens.append(token)
    #     tags.append(tag)
    path = 'data/CD/' + split + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    tokens = [t['tokens'] for t in data]
    tags = [t['tags'] for t in data]

    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
    labels = encode_tags(tags, encodings, cau_tag2id)
    encodings.pop("offset_mapping")

    dataset = TransformersDataset(encodings, labels)

    return dataset, tags


def prepare_number_classification_data(no_sep, split, tokenizer, predict_path=None):
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
        path = 'data/NC/' + split + '.json'

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    sentences = [t['paragraph'] for t in raw]
    labels = [number_tag2id[t['category']] for t in raw]
    offset_start = [int(t['offset_start']) for t in raw]
    offset_end = [int(t['offset_end']) for t in raw]
    numbers = [str(t['target_num']) for t in raw]

    texts = []
    for i in range(len(sentences)):
        if no_sep:
            sent = sentences[i]
            sent = sent[:offset_start[i]] + '<' + numbers[i] + '>' + sent[offset_end[i]:]
        else:
            sent = sentences[i] + '[SEP]' + numbers[i]
        texts.append(sent)

    texts, labels = filter_nc_encoding(texts, labels, tokenizer, 512)
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = TransformersDataset(encodings, labels)

    return dataset


from datasets import load_metric
from sklearn.metrics import mean_squared_error
metric = load_metric("code/metrics/accuracy/accuracy.py")

def compute_metrics_token_classification_f1_srl(eval_pred):
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
            if 0 <= tok <= len(srl_unique_tags):
                pred_label.append(srl_id2tag[predictions[i][j]])
                tag.append(srl_id2tag[l[j]])
        ner_labels.append(pred_label)
        tags.append(tag)
    from seqeval.metrics import f1_score
    return {'accuracy': f1_score(tags, ner_labels, average='macro')}


def compute_metrics_token_classification_f1_cau(eval_pred):
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
                tag.append(cause_tags[l[j]])
        ner_labels.append(pred_label)
        tags.append(tag)
    from seqeval.metrics import accuracy_score
    return {'accuracy': accuracy_score(tags, ner_labels)}


def compute_metrics_sequence_classification(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred, tuple):
        logits = pred[0]
    else:
        logits = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_sequence_regression(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred, tuple):
        logits = pred[0]
    else:
        logits = pred
    predictions = logits
    return {'accuracy': 1 - mean_squared_error(labels, predictions, squared=False)}


def compute_gradients(model):
    gradients = {}

    base_model = model.bert
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is not None and name not in gradients:
                gradients[name] = torch.clone(parameter.grad)

    return gradients


def prepare_data_for_prediction(args, task, tokenizer, predict_path=None):
    if task == 'asa':
        pred_dataset = prepare_aspect_sentiment_data(args.no_sep, 'test', tokenizer, predict_path)
    elif task == 'fpb':
        pred_dataset = prepare_FPB_data('test', tokenizer)
    elif task == 'na':
        pred_dataset = prepare_number_attachment_data(args.no_sep, 'test', tokenizer, predict_path)
    elif task == 'nc':
        pred_dataset = prepare_number_classification_data(args.no_sep, 'test', tokenizer, predict_path)
    elif task == 'srl':
        pred_dataset, pred_tags = prepare_semantic_parsing_data('test', tokenizer)
    elif task == 'cau':
        pred_dataset, pred_tags = prepare_fin_causal_data('test', tokenizer)
    else:
        raise ValueError("task not defined.")

    if task in ['srl', 'cau']:
        return pred_dataset, pred_tags
    else:
        return pred_dataset


def load_single_model_from_mtl_system(args, path, device):
    assert(args.tgt_task in ['asa', 'fpb', 'na', 'srl', 'nc', 'cau'])
    state_dict = torch.load(path + '/pytorch_model.bin', map_location=device)
    state_dict_to_load = collections.OrderedDict()

    for k, v in state_dict.items():
        if k.startswith("taskmodels_dict." + args.tgt_task + "."):
            if "bert.pooler" in k and not args.pool:
                continue
            new_key = k.replace("taskmodels_dict." + args.tgt_task + ".", "")
            state_dict_to_load[new_key] = v

    if args.pal:
        config_dict = {
            'asa': BertForSequenceClassificationAttnPalConfig.from_pretrained(args.model, num_labels=1,
                                                                              layer_attn=args.layer_attn,
                                                                              soft=args.soft, pool=args.pool,
                                                                              arch=args.pal_arch,
                                                                              tuning_size=args.tunning_size),
            'na': BertForSequenceClassificationAttnPalConfig.from_pretrained(args.model, num_labels=2,
                                                                             layer_attn=args.layer_attn, soft=args.soft,
                                                                             pool=args.pool, arch=args.pal_arch,
                                                                             tuning_size=args.tunning_size),
            'fpb': BertForSequenceClassificationAttnPalConfig.from_pretrained(args.model, num_labels=3,
                                                                              layer_attn=args.layer_attn,
                                                                              soft=args.soft, pool=args.pool,
                                                                              arch=args.pal_arch,
                                                                              tuning_size=args.tunning_size),
            'srl': BertForTokenClassificationAttnPalConfig.from_pretrained(args.model, num_labels=len(srl_unique_tags),
                                                                           layer_attn=args.layer_attn, soft=args.soft,
                                                                           pool=args.pool, arch=args.pal_arch,
                                                                           tuning_size=args.tunning_size),
            'nc': BertForTokenClassificationAttnPalConfig.from_pretrained(args.model, num_labels=len(number_labels),
                                                                          layer_attn=args.layer_attn, soft=args.soft,
                                                                          pool=args.pool, arch=args.pal_arch,
                                                                          tuning_size=args.tunning_size),
            'cau': BertForTokenClassificationAttnPalConfig.from_pretrained(args.model, num_labels=5,
                                                                           layer_attn=args.layer_attn, soft=args.soft,
                                                                           pool=args.pool, arch=args.pal_arch,
                                                                           tuning_size=args.tunning_size),
        }

        model_dict = {
            'asa': BertForSequenceClassificationAttnPal.from_pretrained(args.model, config=config_dict['asa']),
            'na': BertForSequenceClassificationAttnPal.from_pretrained(args.model, config=config_dict['na']),
            'fpb': BertForSequenceClassificationAttnPal.from_pretrained(args.model, config=config_dict['fpb']),
            'srl': BertForTokenClassificationAttnPal.from_pretrained(args.model, config=config_dict['srl']),
            'nc': BertForSequenceClassificationAttnPal.from_pretrained(args.model, config=config_dict['nc']),
            'cau': BertForTokenClassificationAttnPal.from_pretrained(args.model, config=config_dict['cau']),
        }
    else:
        config_dict = {
            'asa': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=1,
                                                                           layer_attn=args.layer_attn, pool=args.pool),
            'na': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=2,
                                                                          layer_attn=args.layer_attn, pool=args.pool),
            'fpb': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=3,
                                                                           layer_attn=args.layer_attn, pool=args.pool),
            'srl': BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=len(srl_unique_tags),
                                                                        layer_attn=args.layer_attn, pool=args.pool),
            'nc': BertForSequenceClassificationAttnConfig.from_pretrained(args.model, num_labels=len(number_labels),
                                                                          layer_attn=args.layer_attn, pool=args.pool),
            'cau': BertForTokenClassificationAttnConfig.from_pretrained(args.model, num_labels=5,
                                                                        layer_attn=args.layer_attn, pool=args.pool),
        }

        model_dict = {
            'asa': BertForSequenceClassificationAttn.from_pretrained(args.model, config=config_dict['asa']),
            'na': BertForSequenceClassificationAttn.from_pretrained(args.model, config=config_dict['na']),
            'fpb': BertForSequenceClassificationAttn.from_pretrained(args.model, config=config_dict['fpb']),
            'srl': BertForTokenClassificationAttn.from_pretrained(args.model, config=config_dict['srl']),
            'nc': BertForSequenceClassificationAttn.from_pretrained(args.model, config=config_dict['nc']),
            'cau': BertForTokenClassificationAttn.from_pretrained(args.model, config=config_dict['cau']),
        }

    model = filter_dict([args.tgt_task], model_dict)[args.tgt_task]
    model.load_state_dict(state_dict_to_load)

    return model
