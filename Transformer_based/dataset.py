import random
import torch
from torch.utils.data import Dataset

#from transformers import BertTokenizerFast
import pandas as pd
import pdb


class SentenceDataset(Dataset):

    def __init__(self, path, vocab, test=False):
        self.data = pd.read_csv(path)
        self.vocab = vocab
        self.test = test
        #self.tokenizer = BertTokenizerFast.from_pretrain('bert-base-chinese')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = {}
        fail_sent = self.data.iloc[index]['Question'].split(',')
        tokenize_fail_sent = [self.vocab[pos] for pos in fail_sent ]
        #tokenize_fail_sent = self.tokenizer.encode(fail_sent, add_special_token=True)
        data['pos_sent'] = tokenize_fail_sent
        if self.test == False:
            labels = self.data.iloc[index]['Label'][1:-1].split(',') # index 0 and -1 for '[' and ']'
            labels = list(map(float, labels))
            data['labels'] = labels

        return data


def pred_collate_fn(datas):
    batch = {}
    sent_len = [len(data['pos_sent']) for data in datas]
    batch['length'] = torch.tensor(sent_len)
    max_sent_len = max(sent_len)
    batch['fail_sent'] = torch.tensor([
        pad_to_len(data['pos_sent'], max_sent_len)
        for data in datas]
    )

    mask = torch.zeros(batch['fail_sent'].shape)
    for i, length in enumerate(sent_len):
        mask[i][:length] = 1

    batch['mask'] = (mask == 1)

    return batch


def collate_fn(datas):
    batch = {}
    sent_len = [len(data['pos_sent']) for data in datas]
    batch['length'] = torch.tensor(sent_len)
    max_sent_len = max(sent_len)
    batch['fail_sent'] = torch.tensor([
        pad_to_len(data['pos_sent'], max_sent_len)
        for data in datas]
    )

    batch['labels'] = torch.tensor([
        pad_to_len(data['labels'], max_sent_len, padding=0)
        for data in datas
    ])

    mask = torch.zeros(batch['fail_sent'].shape)
    for i, length in enumerate(sent_len):
        mask[i][:length] = 1

    batch['mask'] = (mask == 1)

    return batch


def pad_to_len(arr, padded_len, padding=0):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    # TODO

    if len(arr)>padded_len:
        return arr[0:padded_len]
    else:
        arr_padded = []
        arr_padded = arr + [padding]*(padded_len - len(arr))
        return arr_padded
