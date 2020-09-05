from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import pandas as pd
from collections import defaultdict
#from add_distribution import check_fail_tractable
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm, trange
import numpy as np
import re
import pickle as pkl
import pdb

#data_utils.download_data_gdown("./") # gdrive-ckip
def split_comma(pos_list):
    size = len(pos_list)
    idx_list =[idx+1 for idx, val in enumerate(pos_list) if val == 'COMMACATEGORY']
    if len(idx_list) == 0:
        idx_list = [size]
    res = [pos_list[i:j] for i,j in 
            zip([0]+idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
    
    return res


def _lcs_dp(a, b):
    """ compute the len dp of lcs"""
    dp = [[0 for _ in range(0, len(b)+1)]
          for _ in range(0, len(a)+1)]
    res = [['' for _ in range(len(b)+1)] for _ in range(len(a)+1)]
    # dp[i][j]: lcs_len(a[:i], b[:j])
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                res[i][j] = res[i-1][j-1] + ',' + str(i-1)
            else:
                if dp[i-1][j] > dp[i][j-1]:
                    dp[i][j] = dp[i-1][j]
                    res[i][j] = res[i-1][j]
                else:
                    dp[i][j] = dp[i][j-1]
                    res[i][j] = res[i][j-1]
    return dp, res

def _lcs_len(a, b):
    """ compute the length of longest common subsequence between a and b"""
    dp, res = _lcs_dp(a, b)
    return dp[-1][-1], res[-1][-1]

def compute_rouge_l(output, reference, mode='f'):
    """ compute ROUGE-L for a single pair of summary and reference
    output, reference are list of words
    """
    assert mode in list('fpr')  # F-1, precision, recall
    lcs, res = _lcs_len(output, reference)
    if lcs == 0:
        score = 0.0
    else:
        precision = lcs / len(output)
        recall = lcs / len(reference)
        f_score = 2 * (precision * recall) / (precision + recall)
        if mode == 'p':
            score = precision
        if mode == 'r':
            score = recall
        else:
            score = f_score
    return score, res


def available_data_num():

    df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
    good_pattern = []
    for index, row in df.iterrows():
        pattern = row['Matched_Frame_Sequential'].split(':')
        if row['Result'] == 'Good':
            for p in pattern:
                if p not in good_pattern:
                    good_pattern.append(p)

    count = 0
    for index, row in df.iterrows():
        pattern = row['Matched_Frame_Sequential'].split(':')
        if row['Result'] == 'Fail':
            for p in pattern:
                if p not in good_pattern or p == 'No Match SemanticPattern':
                    count += 1
                    break
    print(len(df))
    print(count)
    print(len(df) - count)




#generate predict data for NN model
def generate_predict_data():
    ''' 
    這個函數是為了將要被預測的句子轉成詞性序列，接受的格式為小學數學輸出的ALL.csv
    '''
    df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
    fail_sentence_list = []
    mapping = []
    answers = []
    count = 0
    sp = 0
    good_pattern = []
    for index, row in df.iterrows():
        pattern = row['Matched_Frame_Sequential'].split(':')
        if row['Result'] == 'Good':
            for p in pattern:
                if p not in good_pattern:
                    good_pattern.append(p)
    
    for index, row in df.iterrows():
        sent = row['Question'].replace("(1)", "").replace("(2)", "")
        pattern = row['Matched_Frame_Sequential'].split(':')
        split_sent = re.split('？|，|。|：|:', sent)
        if split_sent[-1] == '':
            split_sent = split_sent[: -1]
        if len(split_sent) != len(pattern) or sum([0 if p in good_pattern else 1 for p in pattern])>0:
            sp += 1
            continue
        
        if row['Result'] == 'Fail':
            fail_sentence_list += split_sent
            mapping += [count for _ in range(len(split_sent))]
            answers.append(row['Answer'])
            count += 1

    ws = WS("./data")
    pos = POS("./data")
    fail_word_sent_list = ws(fail_sentence_list)
    fail_pos_sent_list = pos(fail_word_sent_list)
    out = {'Question':[], 'Mapping':[], 'Original':[]}

    for i in range(len(fail_pos_sent_list)):
        out['Question'].append(','.join(fail_pos_sent_list[i]))
        out['Original'].append(','.join(fail_word_sent_list[i]))
        out['Mapping'].append(mapping[i])

    out_df = pd.DataFrame.from_dict(out)
    out_df.to_csv('test.csv')
    pdb.set_trace()
    with open('answer.txt', 'w') as file:
        for ans in answers:
            file.write(str(ans)+'\n')
    

# for NN based
def generate_label_2():
    ''' 
    這個函數是為了將要被預測的句子轉成模型訓練時會用到的資料
    文檔 input : 3顆總共糖果100元
    模型接受的 input(詞性序列) : Neu', 'Nf', 'Na', 'Da', 'Neu', 'Nf'
    output : 1 1 0 0 1 1 1
    '''
    df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
    good_sentence_list = []
    fail_sentence_list = []
    sp = 0
    no_match_num = 0
    all_sent_num = 0
    trash = ['用直式算算看','用分數記記看','再算算看','用直式做做看','用乘法算式記記看','用乘法算式算算看',\
    '用乘法算算看','用除法直式算算看','用連加算式算算看','用算式記記看','先紀錄問題再算算看','先算算看',\
    '再用除法算式記記看','在定位板上用直式算算看','填填看','算算看','寫出乘法算式算算看','寫出算式做做看',\
    '寫出算式算算看','用一個算式記記看']

    for index, row in df.iterrows():

        sent = row['Question'].replace("(1)", "").replace("(2)", "")
        pattern = row['Matched_Frame_Sequential'].split(':')
        match_index = [i for i, match in enumerate(pattern) if match != 'No Match SemanticPattern']
        match_num = [1 if match == 'No Match SemanticPattern' else 0 for match in pattern] 
        no_match_num += sum(match_num)
        all_sent_num += len(pattern)
        split_sent = re.split('？|，|。|：', sent)

        if split_sent[-1] == '':
            split_sent = split_sent[: -1]

        if len(split_sent) != len(pattern):
            sp += 1
            continue

        match_sent = [split_sent[ind] for ind in match_index if split_sent[ind] not in trash] 
        if row['Result'] == 'Good':
            good_sentence_list += match_sent
        if row['Result'] == 'Fail':
            fail_sentence_list += split_sent

    print(no_match_num)
    print(all_sent_num)
    print(sp)
    ws = WS("./data")
    pos = POS("./data")
    good_word_sent_list = ws(good_sentence_list)
    good_pos_sent_list = pos(good_word_sent_list)

    fail_word_sent_list = ws(fail_sentence_list)
    fail_pos_sent_list = pos(fail_word_sent_list)


    del(ws)
    del(pos)

    pattern_dict = []
    golden_pattern = []

    for pos in good_pos_sent_list:
        pos_str = ','.join(pos)
        if pos_str not in golden_pattern:
            golden_pattern.append(pos_str)
        for p in pos:
            if p not in pattern_dict:
                pattern_dict.append(p)
    for pos in fail_pos_sent_list:
        for p in pos:
            if p not in pattern_dict:
                pattern_dict.append(p)
    
    vocab = {'pad':0}
    index = 1
    for pos in pattern_dict:
        if pos not in vocab:
            vocab[pos] = index
            index += 1
    print('The number of golden Pattern is {}'.format(len(golden_pattern)))

    #prob = [0.13, 0.13, 0.14, 0.17, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]
    #prob = [0.17, 0.16, 0.17, 0.18, 0.12, 0.06, 0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.01, 0.005]
    prob = [0.35, 0.18, 0.14, 0.11, 0.09, 0.05, 0.04, 0.02, 0.013, 0.007]
    print(sum(prob))
    insert_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    pattern_num = len(golden_pattern)
    sythesis_data = {'Question': [], 'Label':[]}
    sythesis_data_valid = {'Question':[], 'Label':[]}
    for i in trange(500000):
        sample_idx = np.random.randint(pattern_num)
        sample = golden_pattern[sample_idx]
        k = np.random.choice(insert_num, 1, p=prob)
        while True:
            flag = False
            spos = list(sample.split(','))
            sids = []
            insert_ids = sorted(np.random.randint(len(spos), size=k))
            insert_pos = np.random.choice(pattern_dict, k, replace=False)
            for j, ids in enumerate(insert_ids):
                sids.append(ids)
                spos.insert(ids, insert_pos[j])
                if ids+1<len(spos) and ids-1>=0:
                    if spos[ids+1] == insert_pos[j] or spos[ids-1] == insert_pos[j]:
                        flag = True
                        break
            if flag:
                continue
            if ','.join(spos) not in golden_pattern:
                label = np.ones(len(spos)).tolist()
                for ids in insert_ids:
                    label[ids] = 0.0
                sythesis_data['Question'].append(','.join(spos))
                sythesis_data['Label'].append(label)
                break
            else:
                label = np.ones(len(spos)).tolist()
                sythesis_data['Question'].append(','.join(spos))
                sythesis_data['Label'].append(label)
                break

    train = pd.DataFrame.from_dict(sythesis_data)
    train.to_csv('train.csv', encoding='utf-8')
            
    for i in trange(50000):
        sample_idx = np.random.randint(pattern_num)
        sample = golden_pattern[sample_idx]
        k = np.random.choice(insert_num, 1, p=prob)
        while True:
            flag = False
            spos = list(sample.split(','))
            sids = []
            insert_ids = sorted(np.random.randint(len(spos), size=k))
            insert_pos = np.random.choice(pattern_dict, k, replace=False)
            for j, ids in enumerate(insert_ids):
                sids.append(ids)
                spos.insert(ids, insert_pos[j])
                if ids+1<len(spos) and ids-1>=0:
                    if spos[ids+1] == insert_pos[j] or spos[ids-1] == insert_pos[j]:
                        flag = True
                        break
            if flag:
                continue
            if ','.join(spos) not in golden_pattern:
                label = np.ones(len(spos)).tolist()
                for ids in insert_ids:
                    label[ids] = 0.0
                sythesis_data_valid['Question'].append(','.join(spos))
                sythesis_data_valid['Label'].append(label)
                break
            else:
                label = np.ones(len(spos)).tolist()
                sythesis_data_valid['Question'].append(','.join(spos))
                sythesis_data_valid['Label'].append(label)
                break
            
    valid = pd.DataFrame.from_dict(sythesis_data_valid)
    valid.to_csv('valid.csv', encoding='utf-8')

    with open('vocab.pkl', 'wb') as file:
        pkl.dump(vocab, file)


def rouge_n_only():

    '''
        input filtering 採用Rouge-N的方法

    '''
    df = pd.read_csv('batch_all_2.csv', encoding='utf-8')
    #df = pd.read_csv('might_tractable.csv', encoding='utf-8')
    cut_length = []
    fail_questions = []
    good_sentence_list = []
    fail_sentence_list = []
    mapping = []
    ans = []
    sp = 0
    count = 0
    good_pattern = []
    for index, row in df.iterrows():
        pattern = row['Matched_Frame_Sequential'].split(':')
        if row['Result'] == 'Good':
            for p in pattern:
                if p not in good_pattern:
                    good_pattern.append(p)
    answers = []
    for index, row in df.iterrows():
        sent = row['Question'].replace("(1)", "").replace("(2)", "")
        pattern = row['Matched_Frame_Sequential'].split(':')
        match_index = [i for i, match in enumerate(pattern) if match != 'No Match SemanticPattern']
        result = row['Result']
        split_sent = re.split('？|，|。|：|:', sent)

        if split_sent[-1] == '':
            split_sent = split_sent[: -1]

        if len(split_sent) != len(pattern) or sum([0 if p in good_pattern else 1 for p in pattern])>0:
            sp += 1
            continue

        match_sent = [split_sent[ind] for ind in match_index]
        if result == 'Good':
            good_sentence_list += match_sent
        else:
            fail_sentence_list += split_sent
            mapping += [count for _ in range(len(split_sent))]
            answers.append([len(fail_sentence_list), row['Answer']])
            count += 1
            ans.append(row['Answer'])

    with open('fail_sentence.txt', 'w') as file:
        for sent in fail_sentence_list:
            file.write(sent+'\n')
    with open('answer_mapping.txt', 'w') as file:
        for maps, answer in answers:
            file.write(str(maps)+'\t'+str(answer)+'\n')
    #pdb.set_trace()
    print(sp)
    ws = WS("./data")
    pos = POS("./data")
    good_word_sent_list = ws(good_sentence_list)
    good_pos_sent_list = pos(good_word_sent_list)

    fail_word_sent_list = ws(fail_sentence_list)
    fail_pos_sent_list = pos(fail_word_sent_list)

    del(ws)
    del(pos)

    rewrite_sents = []
    for i in trange(len(fail_pos_sent_list)):
        fail_pos = fail_pos_sent_list[i]
        fail_word = fail_word_sent_list[i]
        lcs_score = [compute_rouge_l(fail_pos, gp, mode='r')[0] for gp in good_pos_sent_list]
        lcs_index = [compute_rouge_l(fail_pos, gp, mode='r')[1] for gp in good_pos_sent_list]
        sorted_index = sorted(lcs_index, key=lambda x:len(x.split(',')[1:]), reverse=True)
        rewrt = [fail_word[int(ind)] for ind in sorted_index[0].split(',')[1:]]
        cut_length.append(len(fail_word) - len(sorted_index[0].split(',')[1:]))
        if len(fail_word) - len(sorted_index[0].split(',')[1:]) > 0:
            pdb.set_trace()
        rewrite_sents.append(rewrt)
    #pdb.set_trace()
    assert len(rewrite_sents) == len(fail_pos_sent_list) == len(mapping)

    plt.hist(cut_length, histtype='stepfilled', alpha=0.3, bins=list(set(cut_length)))
    plt.savefig('cut_length_rouge_o.png')
    output = {}

    for i in range(len(rewrite_sents)):
        if mapping[i] not in output:
            output[mapping[i]] = [rewrite_sents[i]]
        else:
            output[mapping[i]].append(rewrite_sents[i])
    
    assert len(ans) == len(output)

    with open('batch_all_rouge.txt', 'w') as file:
        for index, sent_list in output.items():
            out = ""
            for sent in sent_list:
                out += ''.join(sent) + '，'
            file.write(out[:-1]+'？\t'+ ans[index]+'\n')


def rewrite_function(lcs_res, word_list):
    '''
        此方法為比對結果對句子做改寫
        lcs_res : 比對後要留下的word ids list
        word_list : 句子經過斷詞後產生的list
    '''
    indexes = lcs_res.split(',')[1:]
    return [word_list[int(idx)] for idx in indexes]


if __name__ == '__main__':
    '''
        隨意call function
    '''
    rouge_n_only()
    