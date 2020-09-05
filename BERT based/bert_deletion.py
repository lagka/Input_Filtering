import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer, BertModel
from ckiptagger import data_utils, WS, POS
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from similiar_sentence import TransformerModel

import csv
import pandas as pd
import re
import pickle as pkl
from tqdm import tqdm, trange
from for_experiment import prune

import pdb


beam_size = 3


def rewrite_function(cut_word, l1, l2):
    previous_sent = ''
    for j in range(len(cut_word)):
        if j not in l1+l2:
            previous_sent += cut_word[j]
        elif j in l1:
            previous_sent += len(cut_word[j]) * '#'
        else:
            previous_sent = previous_sent

    return previous_sent


class InputReduction:
    
    def __init__(self, cut_word, redundant_word_ids, distance, sent):
        self.redundant_word_ids = redundant_word_ids
        self.cut_word = cut_word
        self.distance = distance
        self.sent = sent

sp = 0
df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
ws = WS("./data")
pos = POS("./data")
good_sent_list = []
fail_sent_list = []
mapping = []
#threshold = 4.89 # 蘋果重幾公斤 , 蘋果共重幾公斤 4.1503 mean, cls 6.1328 , 4.89 欣欣麵包房 麵包房 bert
threshold = 0.24# 3.4578 蘋果(共)重幾公斤, 4.3946 (欣欣)麵包坊用去4公斤, 3.2563 (是)用去幾包麵粉, 2.7759 160(瓶)牛奶可裝成幾箱又幾包？
#3.3119 8包(共)有多少張
count = 0
ans = []

analysis = []
good_pattern = []
for index, row in df.iterrows():
    pattern = row['Matched_Frame_Sequential'].split(':')
    if row['Result'] == 'Good':
        for p in pattern:
            if p not in good_pattern:
                good_pattern.append(p)

all_pattern_list_g = []
all_pattern_list_f = []
for index, row in df.iterrows():
    
    sent = row['Question'].replace('？','？@')
    if sent == '寫出算式算算看：一盒冰棒有5枝，家裡有5盒，吃了7枝，剩下幾枝冰棒？':
        pdb.set_trace()
    pattern = row['Matched_Frame_Sequential'].split(':')
    split_sent = re.split('@|，|。|：|:', sent)
    match_index = [i for i, match in enumerate(pattern) if match != 'No Match SemanticPattern']
    if split_sent[-1] == '':
        split_sent = split_sent[: -1]
    if len(split_sent) != len(pattern) or sum([0 if p in good_pattern else 1 for p in pattern])>0:
        sp += 1
        if len(split_sent) != len(pattern):
            print(split_sent)
        continue

    if row['Result'] == 'Good':
        match_sent = [split_sent[ind] for ind in match_index]
        good_sent_list += match_sent
        all_pattern_list_g += pattern

        '''
        fail_sent_list += split_sent
        all_pattern_list_f += pattern
        mapping += [count for _ in range(len(split_sent))]
        count += 1
        ans.append(row['Answer'])
        '''

    if row['Result'] == 'Fail':
        fail_sent_list += split_sent
        all_pattern_list_f += pattern
        mapping += [count for _ in range(len(split_sent))]
        count += 1
        ans.append(row['Answer'])

print(sp)
good_word_list = ws(good_sent_list)
good_pattern_list = pos(good_word_list)


'''
所有可處理的句子的詞性見一個字典
'''
golden_pos = {}
for i in range(len(good_pattern_list)):
    patt, ps = all_pattern_list_g[i], good_pattern_list[i]
    if patt not in golden_pos:
        golden_pos[patt] = [ps]
    else:
        if ps not in golden_pos[patt]:
            golden_pos[patt].append(ps)

fail_pattern_list = pos(fail_sent_list)

rewrite_sents = []
beams = []
#model_name = 'hfl/chinese-roberta-wwm-ext-large'
#tokenizer_wmm = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#robert_wmm = BertModel.from_pretrained(model_name).to(device)
#device = torch.device('cpu')
model_name = 'distiluse-base-multilingual-cased'
model = SentenceTransformer(model_name).to(device)
cut_length = []
with open('test.txt', 'w') as file:
    for sent in fail_sent_list:
        file.write(sent+'\n')


'''
底下是bert based input filtering加上beam search
'''
for k in trange(len(fail_sent_list)):
    flag = False
    sent, patt = fail_sent_list[k].replace('總共','').replace('一共','').replace('共',''), all_pattern_list_f[k]
    word = ws([sent])[0]
    with torch.no_grad():
        #sent_emb = robert_wmm(torch.tensor(tokenizer_wmm.encode(sent, add_special_token=True)).\
        #                                unsqueeze(0).to(device))[0][0].mean(dim=0)
        sent_emb = torch.tensor(model.encode(sent))
    if '算算看' in sent or '做做看' in sent or '記記看' in sent or '寫出' in sent:
        rewrite_sents.append('')
        cut_length.append(len(sent))
        continue

    if pos([ws([sent])[0]])[0] in golden_pos[patt]:
        rewrite_sents.append(sent)
        continue

    redundant_word_ids = []
    save_digits_location = [[x.group(), x.start(), x.end()]\
                            for x in re.finditer(r"[-+]?\d*[\+]*\(*\d*/*\.*\d+\)*", sent)]
    #cut_sent = [''.join((re.sub(r'[-+]?\d*[\+]*\(*\d*\/*\.*\d+\)*', '777', sent)))]
    cut_sent = [''.join((re.sub(r'\d*[\+]*\(*\d*\/+\.*\d+\)*', '7', sent)))]
    ori_sent = ws([sent])[0]
    cut_word = ws(cut_sent)[0]
    cut_pos = pos([cut_word])[0]
    save_pos_location = [i for i in range(len(cut_pos)) if cut_pos[i] == 'Nf'\
                         or (cut_pos[i] == 'Neu' and cut_word[i] == '幾')\
                        or cut_pos[i] == 'V_2' or cut_pos[i] == 'Nh' or cut_word[i] == '和' or cut_pos[i] =='Na']
    save_cut_digits_location = [i for i in range(len(cut_word)) if re.match(r'\d*[\+]*\(*\d*\/*\.*\d+\)*', cut_word[i]) is not None]
    chinese_one_location = [i for i in range(len(cut_word)) if cut_word[i] == '一']
    beams.append(InputReduction(list(cut_word), redundant_word_ids, 0.0, sent))
    left_index = [i for i in range(len(cut_word))\
                if i not in (save_pos_location+save_cut_digits_location+chinese_one_location)]

    while True:
        next_beams = []
        previous_beams = list(beams)
        while len(beams) > 0:
            inp = beams.pop()
            cut_word, redundant_word_ids = inp.cut_word, inp.redundant_word_ids
            candidate = [] 
            candidate_sent = []
            previous_sent = rewrite_function(cut_word, redundant_word_ids, save_cut_digits_location)
            for d, s, e in save_digits_location:
                previous_sent = previous_sent[:s] + d + previous_sent[s:]
            with torch.no_grad():
                '''
                previous_sent_hidden = robert_wmm(torch.tensor(tokenizer_wmm.encode(previous_sent.replace('#', ''), add_special_token=True)).\
                                        unsqueeze(0).to(device))[0][0]
                previous_sent_hidden = previous_sent_hidden.mean(dim=0)
                '''
                previous_sent_hidden = torch.from_numpy(model.encode(previous_sent.replace('#',''))[0]).unsqueeze(0)
            #previous_sent_hidden = previous_sent_hidden[0]
            mask_ids = []
            for i in left_index:
                mask_id = list(redundant_word_ids)
                if i in mask_id:
                    continue
                mask_id.append(i)
                removed_one_word_sent = rewrite_function(cut_word, mask_id, save_cut_digits_location)
                '''
                removed_one_word_sent = ''.join([cut_word[j] if j not in mask_id else len(cut_word[j]) * '#'\
                                        for j in range(len(cut_word))])
                '''
                for d, s, e in save_digits_location:
                    removed_one_word_sent = removed_one_word_sent[:s] + d + removed_one_word_sent[s:]
                candidate_sent.append(removed_one_word_sent.replace('#',''))
                #nput_ids = torch.tensor(tokenizer_wmm.encode(removed_one_word_sent.replace('#',''), add_special_token=True)).\
                #               unsqueeze(0).to(device)
                with torch.no_grad():
                    #output = robert_wmm(input_ids)[0].mean(dim=1).squeeze(0)
                    output = torch.tensor(model.encode(removed_one_word_sent.replace('#',''))).squeeze(0)
                candidate.append(output) # for bert batch prediction
                mask_ids.append(mask_id)
                '''
                candidate = ['[CLS '] + list(cut_sent) 
                candidate[i+1] = [' [MASK] ']
                '''
                #candidate = torch.tensor(candidate).cuda()
            try:
                candidate = torch.stack(candidate)
            except:
                break
            distance = torch.sqrt(torch.sum(((candidate - previous_sent_hidden) * (candidate - previous_sent_hidden)), dim=1))
            beam = torch.where(distance < threshold)[0].tolist()
            for index in beam:
                with torch.no_grad():
                    #hidden = robert_wmm(torch.tensor(tokenizer_wmm.encode(candidate_sent[index], add_special_token=True)).\
                    #                    unsqueeze(0).to(device))[0][0].squeeze(0).mean(dim=0)
                    hidden = torch.tensor(model.encode(candidate_sent[index]))
                distance_from_original = torch.dist(hidden, sent_emb, 2).item()
                next_beams.append(InputReduction(list(cut_word), mask_ids[index], distance_from_original, candidate_sent[index]))
            
        if len(next_beams)>1:
            beams = sorted(next_beams, key=lambda x:x.distance, reverse=False)[:beam_size]
        elif len(next_beams) == 0:
            rewrite_sents.append(previous_beams[0].sent)
            cut_length.append(len(sent) - len(previous_beams[0].sent))
            if len(sent) - len(previous_beams[0].sent) > 5:
                analysis.append([sent, previous_beams[0].sent])
            break
        else:
            beams = next_beams

        
        for hypothesis in beams:
            if pos([ws([hypothesis.sent])[0]])[0] in golden_pos[patt]:
                rewrite_sents.append(hypothesis.sent)
                cut_length.append(len(sent) - len(hypothesis.sent))
                if len(sent) - len(hypothesis.sent) > 5:
                    analysis.append([sent, hypothesis.sent])
                beams.clear()
                flag = True
                break
        if flag:
            break

output = {}
for i in range(len(rewrite_sents)):
    if rewrite_sents[i] == '':
        continue
    if mapping[i] not in output:
        output[mapping[i]] = [rewrite_sents[i]]
    else:
        output[mapping[i]].append(rewrite_sents[i])
try:
    plt.hist(cut_length, histtype='stepfilled', alpha=0.3, bins=list(set(cut_length)))
except:
    pdb.set_trace()
    plt.hist(cut_length, histtype='stepfilled', alpha=0.3, bins=14)
plt.savefig('cut_length_rouge_bert.png')
#assert len(ans) == len(output)
#pdb.set_trace()
with open('batch_robert_only_2.txt', 'w') as file:
    for index, sent_list in output.items():
        out = ""
        for sent in sent_list:
            out += ''.join(sent) + '，'
        file.write(out[:-1]+'？\t'+ ans[index]+'\n')
        
pdb.set_trace()
with open('analysis.txt', 'w') as file:
    for sent, rewrt in analysis:
        file.write(sent+'\t'+rewrt+'\n')
prune()