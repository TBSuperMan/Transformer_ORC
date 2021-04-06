from torch.utils import data
import numpy as np
import argparse
import dataset
from torch.utils.data import ConcatDataset
from dataset import Batch
from labelsmap import LabelTool
from transformer import Transformer
import torch.nn as nn
import torch
from torchvision import models
from load_save import *
import crossentropy
import torch.optim as optim
import os
import os.path as osp
from logger import get_logger
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

parserArg = argparse.ArgumentParser()

ZH_ALL_train="/home/gmn/datasets/ZH_ALL_train"
ZH_ALL_test="/home/gmn/datasets/ZH_ALL_test"
IIIT5K_3000="/home/gmn/datasets/IIIT5K_3000_filter"
NIPS2014="/home/gmn/datasets/NIPS2014_filter"
batchsize=32
save_path="/home/gmn/Transformer"

# parserArg.add_argument('--trainRoot', default='/home/gmn/datasets/NIPS2014_filter', type=str)
# parserArg.add_argument('--testRoot', default='/home/gmn/datasets/IIIT5K_3000_filter', type=str)
# parserArg.add_argument('--trainRoot', default='D:/Document/DataSet/reg_dataset/NIPS2014', type=str)
# parserArg.add_argument('--testRoot', default='D:/Document/DataSet/reg_dataset/IIIT5K_3000', type=str)
parserArg.add_argument('--imgH', type=int, default=32)
parserArg.add_argument('--imgW', type=int, default=160)
opt = parserArg.parse_args()

logger = get_logger(osp.join("logs", "Transformer.logs"))
writer = SummaryWriter(comment="NRTR")

ZH_ALL_train_lm=dataset.LmdbDataset(ZH_ALL_train)
ZH_ALL_test_lm=dataset.LmdbDataset(ZH_ALL_test)
NIPS2014_lm = dataset.LmdbDataset(NIPS2014)
IIIT5K_3000_lm = dataset.LmdbDataset(IIIT5K_3000)
train_lm=ConcatDataset([ZH_ALL_train_lm,NIPS2014_lm])
test_lm=ConcatDataset([ZH_ALL_test_lm,IIIT5K_3000_lm])

# train_dataset = dataset.LmdbDataset(opt.trainRoot)
train_dataloader = data.DataLoader(train_lm, batch_size=batchsize, shuffle=True, num_workers=2,
                                   drop_last=True, collate_fn=dataset.AlignCollate(opt.imgH))
# test_dataset = dataset.LmdbDataset(opt.testRoot)

labelTool = LabelTool()

model = Transformer(c_model=256, c_feature=1024, layer_name='layer3',dropout=0.1, c_feedforw=1024, voc_len=len(labelTool.voc)+1)

model = model.cuda()
model = nn.DataParallel(model)

# optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adam(model.parameters())
Loss = crossentropy.SeqCrossEntropy()

# 加载模型参数
if os.path.isfile(osp.join(save_path, 'checkpoint.pth.tar')):
    checkpoint = load_checkpoint(osp.join(save_path, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch']
    best_res = checkpoint['best_res']
    logger.info("=> Start iters {}  best res {:.1%}".format(start_epoch, best_res))

global max_acc, acc_data
max_acc = 0.0
acc_data = []


def val():
    global  max_acc, acc_data
    model.eval()
    model_val = model.module
    for p in model.parameters():
        p.requires_grad = False
    test_dataloader = data.DataLoader(test_lm, batch_size=batchsize, shuffle=True,
                                       drop_last=True, collate_fn=dataset.AlignCollate(opt.imgH))
    test_iter = iter(test_dataloader)
    max_iter = min(20, len(test_dataloader))
    n_correct = 0
    for k in range(max_iter):
        images, labels_y, labels, lengths = next(test_iter)
        batch = Batch(images, labels_y, labels)
        images = model_val.img_pos_enc(model_val.featureExtractor(batch.src))
        images = model_val.encoder(images, batch.src_mask)

        batch_size = images.size(0)
        max_len = max(lengths)
        tgt = torch.full((batch_size, max_len+1), labelTool.char2id['PADDING']).cuda()
        tgt[:, 0] = labelTool.char2id['<']
        for l in range(max_len):
            tgt_now = model_val.tgt_embedding(tgt[:, :l+1].long())
            tgt_now = model_val.tgt_pos_enc(tgt_now)
            tgt_mask = torch.ones(batch_size, 1, l+1).cuda()
            x = model_val.decoder(tgt_now, images, batch.src_mask, tgt_mask)
            output = model_val.out_softmax(x)
            _, output = output.max(dim=-1)
            index = output[:, -1]
            tgt[:, l+1] = index

        pred_list = []
        for item in range(batch_size):
            pred_item = []
            temp = tgt[item]
            for i in range(max_len):
                char = labelTool.id2char[temp[i+1].item()]
                if char == '>':
                    break
                pred_item.append(char)
            pred_list.append(''.join(pred_item))

        tgt_list = []
        for item in range(batch_size):
            tgt_item = []
            temp = labels_y[item]
            for i in range(max_len):
                char = labelTool.id2char[temp[i].item()]
                if char == '>':
                    break
                tgt_item.append(char)
            tgt_list.append(''.join(tgt_item))

        for pred, label in zip(pred_list, tgt_list):
            if pred == label:
                n_correct += 1

    for pred, label, n in zip(pred_list, tgt_list, range(10)):
        logger.info('%-20s  ==>  %-20s' % (pred, label))

    acc = n_correct / float(max_iter * batch_size)
    acc_data.append(acc)
    if acc > max_acc:
        max_acc = acc
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_res': max_acc,
        }, True, fpath=osp.join(save_path, 'checkpoint.pth.tar'))
    else:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_res': max_acc,
        }, False, fpath=osp.join(save_path, 'checkpoint.pth.tar'))

    writer.add_scalar('Acc', acc, k)
    logger.info('acc:%f  max_acc:%f' % (acc, max_acc))
    # logger.info('acc_data:', acc_data)

def train_batch():
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    # image  batch * 3 * 32 * max_width
    images, labels_y, labels, lengths = next(train_iter)
    batch = Batch(images, labels_y, labels)
    output = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)
    # output: batch * seq_len * len(voc)
    max_len = max(lengths)
    output = output[:, :max_len]
    loss = Loss.getLoss(output, batch.trg_y, lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del batch.src, batch.trg_y, batch.src_mask, batch.trg_mask
    torch.cuda.empty_cache()
    return loss




for epoch in range(500):
    train_iter = iter(train_dataloader)
    for k in range(len(train_iter)):
        k += 1
        loss = train_batch()
        writer.add_scalar('Loss/train', loss, k)
        logger.info('epoch:%d  iter:%d/%d loss:%f' % (epoch, k,len(train_iter), loss))
        if k % 1000 == 0:
            val()
