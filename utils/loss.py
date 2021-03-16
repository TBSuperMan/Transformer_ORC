import torch.nn.functional as F
import torch

class seqCrossEntropy():
    def __init__(self):
        return

    def getLoss(self, preds, targets, lengths):
        batch_size, whole_len= targets.size(0), targets.size(1)
        #cuda
        mask = torch.zeros(batch_size, whole_len).cuda()
        for i in range(batch_size):
            mask[i, :lengths[i]].fill_(1)
        max_len = max(lengths)
        targets = targets[:, :max_len]
        mask = mask[:, :max_len]
        targets = targets.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        preds = preds.contiguous().view(-1, preds.size(2))
        preds = F.log_softmax(preds, dim=1)
        output = -preds.gather(1, targets.long()) * mask
        output = torch.sum(output)
        output = output / batch_size
        return output
