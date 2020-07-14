#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix

logger = logging.getLogger(__name__)

EOS_ID = 50256

def nltk_BLEU_4(generated, reference):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    BLEUscore[0] = corpus_bleu(reference, generated, weights=(1,0,0,0))*100
    BLEUscore[1] = corpus_bleu(reference, generated, weights=(0,1,0,0))*100
    BLEUscore[2] = corpus_bleu(reference, generated, weights=(0,0,1,0))*100
    BLEUscore[3] = corpus_bleu(reference, generated, weights=(0,0,0,1))*100

    return BLEUscore


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    generated =[]
    reference =[]

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            input_ids, position_ids, token_ids, label_ids, emotion_labels, da_labels, *_ = batch

            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl, lm_logits, emotion_logits, da_logits = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)

            for label_id in label_ids:
                label_id = label_id[label_id!=-1]
                reference.append([tokenizer.decode(label_id).split(' ')])

            _, predicted_ids = torch.max(lm_logits, dim=2)

            for predicted_id in predicted_ids:
                generated.append(tokenizer.decode(predicted_id).split(' '))

    BLEUscore = nltk_BLEU_4(generated, reference)

    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    print(f"\n BLEU 1 {BLEUscore[0]}")
    print(f"\n BLEU 2 {BLEUscore[1]}")
    print(f"\n BLEU 3 {BLEUscore[2]}")
    print(f"\n BLEU 4 {BLEUscore[3]}")


    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)



def get_model_metrics(model, tokenizer, eval_dataloader, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    generated =[]
    reference =[]
    emotion_labels_op, emotion_labels_pred =[] , []
    da_labels_op, da_labels_pred =[] , []

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            input_ids, position_ids, token_ids, label_ids, emotion_labels, da_labels, *_ = batch

            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl, lm_logits, emotion_logits, da_logits = model(input_ids, position_ids, token_ids, label_ids, \
                emotion_labels=emotion_labels, da_labels=da_labels)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)

            for input_id in input_ids:
                reference.append([tokenizer.decode(input_id).split(' ')])

            _, predicted_ids = torch.max(lm_logits, dim=2)

            for predicted_id in predicted_ids:
                generated.append(tokenizer.decode(predicted_id).split(' '))

            _, emo_ids = torch.max(emotion_logits, dim=1)
            _, da_ids = torch.max(da_logits, dim=1)

            emotion_labels_op.extend(emotion_labels.tolist())
            emotion_labels_pred.extend(emo_ids.tolist())

            da_labels_op.extend(da_labels.tolist())
            da_labels_pred.extend(da_ids.tolist())



    BLEUscore = nltk_BLEU_4(generated, reference)

    #print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    print(f"\n BLEU 1 {BLEUscore[0]}")
    print(f"\n BLEU 2 {BLEUscore[1]}")
    print(f"\n BLEU 3 {BLEUscore[2]}")
    print(f"\n BLEU 4 {BLEUscore[3]}")

    print(f"\n F1 EMO {f1_score(emotion_labels_op, emotion_labels_pred, average='macro')}")
    print(f"\n precision_score EMO {precision_score(emotion_labels_op, emotion_labels_pred, average='macro')}")
    print(f"\n recall_score EMO {recall_score(emotion_labels_op, emotion_labels_pred, average='macro')}")
    print(f"\n accuracy_score EMO {accuracy_score(emotion_labels_op, emotion_labels_pred)}")
    print(f"\n confusion_matrix EMO {confusion_matrix(emotion_labels_op, emotion_labels_pred)}")


    print(f"\n F1 DA {f1_score(da_labels_op, da_labels_pred, average='macro')}")
    print(f"\n precision_score DA {precision_score(da_labels_op, da_labels_pred, average='macro')}")
    print(f"\n recall_score DA {recall_score(da_labels_op, da_labels_pred, average='macro')}")
    print(f"\n accuracy_score DA {accuracy_score(da_labels_op, da_labels_pred)}")
    print(f"\n confusion_matrix DA {confusion_matrix(da_labels_op, da_labels_pred)}")


    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)  
