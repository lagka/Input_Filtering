import pdb
from transformers import BertForMaskedLM, BertTokenizer, BertModel
import torch
from ckiptagger import data_utils, construct_dictionary, WS
import torch.nn.functional as F
from tqdm import tqdm
import re
import pandas as pd


#改檔案名，沒啥大作用的function
def prune():
    question, answer = [], []
    with open('batch_robert_only_2.txt', 'r') as file:
        for line in file:
            l = line.split('\t')
            question.append(l[0].replace('？，','？').replace('？？','？'))
            answer.append(l[1])

    assert len(question) == len(answer)

    with open('robert_rule_all.txt', 'w') as file:
        for i in range(len(question)):
            file.write(question[i]+'\t'+answer[i])


#計算句子通順度
def calculate_perplexity(path):
    model_name = 'hfl/chinese-roberta-wwm-ext-large'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    maskedLM_model = BertForMaskedLM.from_pretrained(model_name).cuda()
    sents = []
    perplexity = 0
    with open(path, 'r') as file:
        for line in file:
            question = line.split()[0]
            sents += question.split('，')
    
    description = 'perplexity'
    iter_in_epoch = len(sents)
    trange = tqdm(enumerate(sents),
                total=iter_in_epoch,
                desc=description)
    for j, sent in trange:
        sent = re.sub(r"[-+]?\d*[\+]*\(*\d*/*\.*\d+\)*","8", sent)
        p = 1
        cut_l = len(sent)
        for i in range(len(sent)):
            masked_sent = sent[:i]+' [MASK] ' + sent[i+1:]
            sub_sent = torch.tensor(tokenizer.encode(masked_sent)).unsqueeze(0).cuda()
            segment_tensor = torch.zeros_like(sub_sent).cuda()
            with torch.no_grad():
                prediction = F.softmax(maskedLM_model(sub_sent, segment_tensor)[0], dim=2)
            target = tokenizer.encode(sent[i])[1]
            if sent[i] == '8':
                continue
            try:
                if p * prediction[0][i+1][target] == 0.0:
                    cut_l = i
                    break
            except:
                p = 1
                break
            p *= prediction[0][i+1][target]
        try:
            perplexity+= pow(p, -1/(cut_l))
        except:
            continue
        trange.set_postfix({'Current perplexity':"{}".format(perplexity/(j+1))})
    print(perplexity/len(sents))


if __name__ == '__main__':
    
    a = ['test.txt']
    for path in a:
        calculate_perplexity(path)