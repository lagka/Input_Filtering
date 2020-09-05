#from transformers import BertPreTrainedModel

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from dataset import SentenceDataset, collate_fn, pred_collate_fn
import pickle as pkl
import os

import pdb

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda = True if torch.cuda.is_available() else False


def sequence_loss(logits, targets, mask, xent_fn=None):

    target = targets.masked_select(mask)
    logit = logits.masked_select(mask)
    #pdb.set_trace()
    if xent_fn:
        loss = xent_fn(logit, target).sum()
    return loss


class TransformerModel(nn.Module):

    def __init__(self, ninp, vocab_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, ninp)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.ninp = ninp

        self.linear = nn.Sequential(
            nn.Linear(ninp, ninp//2),
            nn.ReLU(),
            nn.Linear(ninp//2, 1)
        )

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        nn.init.kaiming_normal_(self.emb.weight)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.linear(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_criterion():
    nll = lambda logit, target: F.binary_cross_entropy_with_logits(logit, target, reduce=False,\
                                pos_weight=torch.ones([target.shape[0]]).to(device)*0.75)
    def criterion(logits, targets, masks):
        return sequence_loss(logits, targets, masks, nll)
    return criterion


def get_training_setting():

    with open('vocab.pkl', 'rb') as file:
        vocab = pkl.load(file)
    
    reverse_vocab = dict((v, k) for k, v in vocab.items())

    emsize = 40 # embedding dimension
    nhid = 40 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(emsize, len(vocab), nhead, nhid, nlayers, dropout).to(device)

    lr = 1e-2 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    dataloader = Data.DataLoader(
        dataset = SentenceDataset('train.csv', vocab), batch_size = 64,\
        shuffle = True, collate_fn = collate_fn, num_workers = 4
    )
    val_data = Data.DataLoader(
        dataset = SentenceDataset('valid.csv', vocab), batch_size=64,\
        shuffle = False, collate_fn = collate_fn, num_workers=4
    )
    return model, scheduler, optimizer, dataloader, val_data, reverse_vocab


def train_one_epoch(model, optimizer, train_data, criterion):

    bptt = 64
    model.train()
    total_loss = 0
    start_time = time.time()
    iter_in_epoch = len(train_data)
    description = 'training'
    trange = tqdm(enumerate(train_data),
                total=iter_in_epoch,
                desc=description)
    for i, batch in trange:
        data, targets, masks = batch['fail_sent'], batch['labels'], batch['mask']
        if cuda:
            data, targets, masks = data.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(2), targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        trange.set_postfix({'Total loss':"{0:.6f}".format(total_loss/(i+1))})
        '''
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            
            total_loss = 0
            start_time = time.time()
        '''


def evaluate(epoch, eval_model, data_source, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_source:
            data, targets, masks = batch['fail_sent'], batch['labels'], batch['mask']
            if cuda:
                data, targets, masks = data.to(device), targets.to(device), masks.to(device)
            output = eval_model(data).squeeze(2)
            total_loss += criterion(output, targets, masks).item()
    return total_loss / (len(data_source) - 1)


def predict(path):


    with open('vocab.pkl', 'rb') as file:
        vocab = pkl.load(file)
    reverse_vocab = {v: k for k, v in vocab.items()}
    emsize = 40 # embedding dimension
    nhid = 40# the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 3 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(emsize, len(vocab), nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(path)['model'])

    test_data = Data.DataLoader(
        dataset = SentenceDataset('test.csv', vocab, True), batch_size = 64,\
        shuffle = False, collate_fn = pred_collate_fn, num_workers = 4
    )
    model.eval()
    rewrite_indexes = []
    cut_length = []
    for batch in test_data:
        data, masks, length= batch['fail_sent'], batch['mask'], batch['length']
        if cuda:
            data, masks = data.to(device), masks.to(device)
        output = model(data).squeeze(2)
        probs = (F.sigmoid(output) > 0.6)
        for i in range(data.shape[0]):
            check = ','.join([reverse_vocab[idx.item()] for idx in data[i]])
            if check == 'Nb,P,Neu,Nf,Na,VG,Neu,Nf,Ng':
                pdb.set_trace()
            d, l, p = data[i], length[i], probs[i]
            one_hot = (p == True).tolist()[:l]
            index = [i for i in range(len(one_hot)) if one_hot[i] == 1]
            cut_length.append(len(one_hot)-len(index))
            rewrite_indexes.append(index)
    
    output = {}
    answers = []
    with open('answer.txt', 'r') as file:
        for line in file:
            answers.append(line.split('\n')[0])

    plt.hist(cut_length, histtype='stepfilled', alpha=0.3, bins=list(set(cut_length)))
    plt.savefig('cut_length_rouge_transformer.png')
    df = pd.read_csv('test.csv')
    for i, row in df.iterrows():
        index = rewrite_indexes[i]
        word_list = row['Original'].split(',')
        mapping = row['Mapping']
        sent = [word_list[ind] for ind in index]
        if mapping not in output:
            output[mapping] = [sent]
        else:
            output[mapping].append(sent)
    
    with open('rewrite_index.txt', 'w') as file:
        for key, value in output.items():
            out = ""
            for sent in value:
                out += ''.join(sent)+'，'
            try:
                out = out[:-1] + '？\t' + answers[key] + '\n'
            except:
                pdb.set_trace()
            file.write(out)


def train(model, scheduler, optimizer, train_data, val_data, criterion):
    best_val_loss = float("inf")
    epochs = 40 # The number of epochs
    best_model = None
    if not os.path.exists('./sim_test/'):
        os.makedirs('./sim_test/')

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_one_epoch(model, optimizer, train_data, criterion)
        val_loss = evaluate(epoch, model, val_data, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}|'
                'Lr: {:.5f}'.format(epoch, (time.time() - epoch_start_time),
                val_loss, optimizer.param_groups[0]['lr']))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            
            torch.save({
                'epoch': epoch,
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, './sim_test/'+str(epoch))
            
        else:
            scheduler.step()


if __name__ == '__main__':
    flag = False
    if flag:
        criterion = get_criterion()
        model, scheduler, optimizer, train_data, val_data, reverse_vocab = get_training_setting()
        train(model, scheduler, optimizer, train_data, val_data, criterion)
    else:
        predict('./sim_test/16')