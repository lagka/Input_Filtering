from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm, trange
import re
import matplotlib.pyplot as plt
from ckippy import parse_tree

from ckiptagger_test import compute_rouge_l
import pdb


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False


#根據Label Sequence我個人設計的reduction機制
def hand_craft_reduction(ls):
    
    rules = ['<數量>,<物件>','<長度>,<的>,<物件>','<重量>,<的>,<物件>','<容量>,<的>,<物件>']
    if rules[0] in ls or rules[1] in ls or rules[2] in ls or rules[3] in ls:
        for i in range(len(rules)):
            ls = ls.replace(rules[i], '<物件>')
        return ls
    else:
        return ls

#根據一個句子的Label Sequence來預測是屬於哪個動詞框架
def text_classification():

    sp = 0
    df = pd.read_csv('might_tractable.csv', encoding='utf-8')
    df  = df.loc[df['Result'] == 'Good']
    ground_truth, good_sents, answers = [], [], []
    for index, row in df.iterrows():
        question = row['Question']
        sents = re.split('？|,|，|。|：|:|、', question)
        pattern = row['Matched_Frame_Sequential'].split(':')
        if sents[-1] == '':
            sents = sents[:-1]
        if len(sents) != len(pattern):
            sp += 1
            continue
        ground_truth += pattern
        good_sents += sents
        answers.append([len(good_sents), row['Answer']])

    with open('all_good_sentence.txt', 'w') as file:
        for sent in good_sents:
            file.write(sent+'\n')
    
    with open('answer_mapping.txt', 'w') as file:
        for maps, answer in answers:
            file.write(str(maps)+'\t'+str(answer)+'\n')
    label_sents = []
    sents = []
    count = 0
    unreduce_ls = []
    with open('label_sequence.txt', 'r') as file:
        for line in file:
            split_line = line.split()
            if len(split_line) > 0:
                if split_line[0] == '#':
                    sents.append(split_line[1])
                    reduce_ls = hand_craft_reduction(split_line[2])
                    label_sents.append(reduce_ls)
                    unreduce_ls.append(split_line[2])
    
    predictions = []
    pattern_label_dict, label_pattern_dict, pattern_label_count = {}, {}, {}
    test = []
    for i in range(len(label_sents)):
        ls = label_sents[i]
        gt = ground_truth[i]
        ch = np.random.choice(2, 1, p=[0.9, 0.1])
        if ch == 0 :
            if gt not in pattern_label_dict:
                pattern_label_dict[gt] = [ls]
                pattern_label_count[gt+'#'+ls] = 0
            else:
                if ls not in pattern_label_dict[gt]:
                    pattern_label_dict[gt].append(ls)
                    pattern_label_count[gt+'#'+ls] = 0
                else:
                    pattern_label_count[gt+'#'+ls] += 1
        else :
            test.append([ls, gt, good_sents[i], unreduce_ls[i]])

    for p, lls in pattern_label_dict.items():
        for label in lls:
            if label not in label_pattern_dict:
                label_pattern_dict[label] = p
            else:
                if pattern_label_count[p+'#'+label] > pattern_label_count[label_pattern_dict[label]+'#'+label]:
                    #print("{} : {} : {}".format(p, label, label_pattern_dict[label]))
                    label_pattern_dict[label] = p
                '''
                else:
                    print("2 {} : {} : {}".format(p, label, label_pattern_dict[label]))
                '''
    
    correct = 0
    out_pattern = 0
    for i, (ls, gt, st, uls) in enumerate(test):
        try:
            pred = label_pattern_dict[ls]
        except:
            pred = gt
            label_pattern_dict[ls] = gt
            #print('{} {} {}'.format(ls, gt, st))
        if pred == gt:
            correct += 1

    print('Accuracy : {:.5f} {}'.format((correct)/len(test), len(label_pattern_dict)))
    data = {}
    data['Sentence'] = good_sents
    data['Label Sequence'] = label_sents
    data['UnReduce Label Sequence'] = unreduce_ls
    out = pd.DataFrame.from_dict(data)
    out.to_csv('Label_sent.csv', encoding='utf-8')
    with open('LabelSequence_Pattern.txt', 'w') as file:
        for ls, p in label_pattern_dict.items():
            file.write(ls+'\t'+p+'\n')
    #pdb.set_trace()


# 論文做計算目前可以答對的題目類型，以及無法答對的題目類型數量分布
def gf_question_type():
    
    col_count = 6
    bar_width = 0.2
    index = np.arange(col_count)
    question_from = {}
    good_result_type = {}
    with open('good_result.txt', 'r') as file:
        for line in file:
            line = line.split()
            if line [2] not in good_result_type:
                good_result_type[line[2]] = 1
            else:
                good_result_type[line[2]] += 1
            
            if line[5]+','+line[2] not in question_from:
                question_from[line[5]+','+line[2]] = 1
            else:
                question_from[line[5]+','+line[2]] += 1

    fail_result_type = {}
    fail_question = []
    pdb.set_trace()
    with open('fail_result.txt', 'r') as file:
        for line in file:
            line = line.split()
            if line[1] in fail_question:
                continue
            else:
                fail_question.append(line[1])
            qt = line[3]
            qf = line[6]
            if qt == 'a.m.' or qt == 'p.m.' or qt == '4:30' or qt == '3' or qt == '9:15' or qt == 'PM' or\
                qt == '5' or qt == '時50分' or qt == '8':
                qt = line[4]
                qf = line[7]
            if qt not in fail_result_type:
                fail_result_type[qt] = 1
            else:
                fail_result_type[qt] += 1
            
            if qf+','+qt not in question_from:
                question_from[qf+','+qt] = 1
            else:
                question_from[qf+','+qt] += 1

    pdb.set_trace()
    
    question_type = [key for key, _ in good_result_type.items()]
    good_question_type_num = [value for _, value in good_result_type.items()]

    fail_question_type_num = [fail_result_type[qt] for qt in question_type]
    
    del(fail_result_type['a.m.'])
    del(fail_result_type['p.m.'])

    good = plt.bar(index, 
                good_question_type_num,
                bar_width,
                alpha=.4,
                label="Good Question")
    fail = plt.bar(index+0.2,
                fail_question_type_num,
                bar_width,
                alpha=.4,
                label="Fail Question")
    
    createLabels(good)
    createLabels(fail)
    plt.ylabel("Question num")
    plt.xlabel("Question type")
    plt.title("Distribution")
    plt.xticks(index+.4 / 2 , question_type)
    plt.legend() 
    plt.grid(True)
    plt.savefig('question_type.png')
    pdb.set_trace()



#用來畫出各個方法刪減字數的分佈圖，除了Label Sequence
def _remove_word_num_distribution_():
    df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
    good_sentence_list = []
    fail_sentence_list = []
    sp = 0
    for index, row in df.iterrows():

        sent = row['Question'].replace("(1)", "").replace("(2)", "")
        pattern = row['Matched_Frame_Sequential'].split(':')
        match_index = [i for i, match in enumerate(pattern) if match != 'No Match SemanticPattern']

        split_sent = re.split('？|，|。|：|:', sent)

        if split_sent[-1] == '':
            split_sent = split_sent[: -1]

        if len(split_sent) != len(pattern):
            sp += 1
            continue

        match_sent = [split_sent[ind] for ind in match_index]
        if row['Result'] == 'Good':
            good_sentence_list += match_sent
        else:
            fail_sentence_list += split_sent


    print(sp)
    ws = WS("./data")
    pos = POS("./data")
    good_word_sent_list = ws(good_sentence_list)
    good_pos_sent_list = pos(good_word_sent_list)

    fail_word_sent_list = ws(fail_sentence_list)
    fail_pos_sent_list = pos(fail_word_sent_list)

    x = []
    counter = defaultdict(int)
    for i in trange(len(fail_pos_sent_list)):
        fail_pos = fail_pos_sent_list[i]
        lcs_score = [compute_rouge_l(fail_pos, gp, mode='r')[0] for gp in good_pos_sent_list]
        lcs_indexes = [compute_rouge_l(fail_pos, gp, mode='r')[1] for gp in good_pos_sent_list]
        fidx = np.argwhere(lcs_score == np.amax(lcs_score))
        sorted_idx = sorted(fidx, key=lambda x:len(lcs_indexes[x.item()].split(',')[1:]), reverse=True)
        idx = sorted_idx[0]
        x += [len(fail_pos) - len(lcs_indexes[idx.item()].split(',')[1:])]
        counter[len(fail_pos) - len(lcs_indexes[idx.item()].split(',')[1:])] += 1
        '''
        for idx in fidx:
            x += [len(fail_pos) - len(lcs_indexes[idx.item()].split(',')[1:])]
            counter[len(fail_pos) - len(lcs_indexes[idx.item()].split(',')[1:])] += 1
        '''
    for k, value in counter.items():
        print("Key is {}, prob : {:.4f}".format(k, value/len(x)))
    #pdb.set_trace()
    plt.hist(x, histtype='stepfilled', alpha=0.3, density=True, bins=13)
    plt.savefig('distribution_tractable.png')

#用來畫出Label Sequence刪減字數的分佈圖
def LS_deletion_distribution():

    df = pd.read_csv('might_tractable.csv', encoding='utf-8')
    dff = df.loc[df['Result'] == 'Good']
    good_sents, mapping, answers = [], [], []
    for index, row in dff.iterrows():
        question = row['Question']
        sents = re.split('？|,|，|。|：|:|、', question)
        if sents[-1] == '':
            sents = sents[:-1]
        good_sents += sents
        answers.append([len(good_sents), row['Answer']])
    
    ls_sent = []
    with open('label_sequence.txt', 'r') as file:
        for line in file:
            ls_sent.append(line.split()[0])

    cut_length = []
    for i in range(len(good_sents)):
        cut_length.append(len(good_sents[i]) - len(ls_sent[i]))
    plt.hist(cut_length, histtype='stepfilled', alpha=0.3, bins=list(set(cut_length)))
    plt.savefig('cut_length_rouge_ls.png')

if __name__ == '__main__':
    #gf_question_type()
    #check_fail_tractable()
    '''
    for _ in range(1):
        text_classification()
    '''
    LS_deletion_distribution()
    #sent_to_parse_tree()
    #_remove_word_num_distribution_()
