import numpy as np
from utils import srl_id2tag, srl_unique_tags
from sklearn.metrics import mean_squared_error



def compute_metrics_token_classification_f1(eval_pred):
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
    return {'macro_f1': f1_score(tags, ner_labels, average='macro')}


def compute_metrics_token_classification_acc(eval_pred):
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
    from seqeval.metrics import accuracy_score
    return {'accuracy': accuracy_score(tags, ner_labels)}


def compute_metrics_sequence_classification(eval_pred):
    pred, labels = eval_pred[0], eval_pred[1]
    if isinstance(pred, tuple):
        logits = pred[0]
    else:
        logits = pred
    predictions = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score
    return {'accuracy': accuracy_score(labels, predictions)}


def compute_metrics_sequence_regression(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred, tuple):
        logits = pred[0]
    else:
        logits = pred
    predictions = logits
    return {'1-RMSE': 1 - mean_squared_error(labels, predictions, squared=False)}