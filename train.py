import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import Labelsmap
from datasets.dataset import Batch
from datasets.dataset import alignCollate
from models.Transformer import make_model
from datasets.dataset import lmdbDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
import os.path as osp
from utils.loss import seqCrossEntropy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,1"

train_path="D:/Document/DataSet/reg_dataset/NIPS2014"
test_path="D:/Document/DataSet/reg_dataset/IIIT5K_3000"

romte_train_path="/home/gmn/datasets/NIPS2014"
romte_test_path="/home/gmn/datasets/IIIT5K_3000"
test_iter=1000
batchsize=32
numworker=0
imgH=48
imgW=160
cuda=True

class NoamOpt:
    """
    Optim wrapper that implements rate.
    动态根据step调整优化器的学习率
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        target=target.long()
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))




def val():
    global max_acc,acc_data
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    val_iter = iter(val_dataloader)
    n_correct = 0
    max_iter = min(20,len(val_dataloader))
    for i in range(max_iter):
        images, labels_y, labels, lengths= next(val_iter)
        batch = Batch(images, labels_y, labels)
        images = model.src_embed(batch.src)
        images = model.encoder(images, batch.src_mask)

        batch_size = images.size(0)
        max_len = max(lengths)
        tgt = torch.full((batch_size, max_len+1), labelTool.char2id['PADDING']).cuda()
        tgt[:, 0] = labelTool.char2id['<']
        for l in range(max_len):
            tgt_now = model.tgt_embed(tgt[:, :l + 1].long())
            tgt_mask = torch.ones(batch_size, 1, l + 1).cuda()
            x = model.decoder(tgt_now, images, batch.src_mask, tgt_mask)
            output = model.generator(x)
            _, output = output.max(dim=-1)
            index = output[:, -1]
            tgt[:, l + 1] = index
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
        print('%-20s  ==>  %-20s' % (pred, label))

    acc = n_correct / float(max_iter * batch_size)
    acc_data.append(acc)
    if acc > max_acc:
        max_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
    print('acc:%f  max_acc:%f' % (acc, max_acc))
    print('acc_data:', acc_data)

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
    loss = loss_weight.getLoss(output, batch.trg_y, lengths)

    model_opt.optimizer.zero_grad()
    loss.backward()
    model_opt.step()

    del batch.src, batch.trg_y, batch.src_mask, batch.trg_mask
    torch.cuda.empty_cache()
    return loss


global max_acc,acc_data
max_acc=0.0
acc_data=[]

if __name__ == '__main__':
    # 日志
    logger = get_logger(osp.join("logs", "NRTR.logs"))
    labelTool = Labelsmap.LabelTool()

    # train_lm = lmdbDataset(train_path)
    # test_lm = lmdbDataset(test_path)
    train_lm = lmdbDataset(romte_train_path)
    test_lm = lmdbDataset(romte_test_path)
    train_dataloader = DataLoader(train_lm, batch_size=batchsize, num_workers=numworker,
                             shuffle=True, pin_memory=True, drop_last=True,
                             collate_fn=alignCollate(imgH=imgH, keep_ratio=False))

    val_dataloader = DataLoader(test_lm, batch_size=batchsize, num_workers=numworker,
                             shuffle=True, pin_memory=True, drop_last=True,
                             collate_fn=alignCollate(imgH=imgH, keep_ratio=False))

    model = make_model(len(labelTool.voc)+1)
    # model.load_state_dict(torch.load('your-pretrain-model-path'))
    model.cuda()

    # 损失函数
    # criterion = LabelSmoothing(size=len(train_lm.char2id), padding_idx=0, smoothing=0.1)
    # criterion.cuda()

    # 损失函数
    loss_weight = seqCrossEntropy()
    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

    #多卡并行
    device_ids=[0, 1,2,3]
    model = nn.DataParallel(model,device_ids=device_ids)
    model=model.module
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    optimizer=optimizer.module

    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,optimizer)

    train_iter = iter(train_dataloader)
    total_loss=0.0
    for i in range(10):
        for k in range(len(train_dataloader)):
            k += 1
            loss = train_batch()
            total_loss += loss
            if k%50 ==0:
                print('epoch:%d  iter:%d loss:%f' % (i, k, total_loss/50))
                total_loss = 0.0
            if k % 1000 == 0:
                val()

