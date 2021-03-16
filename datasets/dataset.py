import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import  transforms
import lmdb
from utils import Labelsmap
import six
from torch.utils.data import DataLoader
import math

labelTool = Labelsmap.LabelTool()

class lmdbDataset(Dataset):
    def __init__(self, root, max_len=100):
        self.env = lmdb.open(root)
        self.txn=self.env.begin(write=False)
        self.nSamples = int(self.txn.get("num-samples".encode()))
        self.txn = self.env.begin()
        self.voc = labelTool.voc
        self.char2id = labelTool.char2id
        self.id2char = labelTool.id2char
        self.max_len = max_len
        for i, c in enumerate(self.voc):
            self.char2id[c] = i + 1
            self.id2char[i + 1] = c

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        img_key = 'image-%09d' % index
        imgbuf = self.txn.get(img_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        label_key = 'label-%09d' % index
        word = self.txn.get(label_key.encode()).decode()
        lens=len(word)+1

        # As pytorch tensor
        label = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate('<' + word):
            label[i] = self.char2id[c]

        label_y = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate(word + '>'):
            label_y[i] = self.char2id[c]

        return img,label_y,label,lens


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, imgs, trg_y, trg, pad=0):
        # print("trg_y.shape",trg_y.shape)
        # print("trg.shape", trg.shape)
        self.src = imgs.cuda() #[batch,channel,h,w]
        #src_mask [batch,1,h/16*w/16]
        # print("imgs.shape ",imgs.shape)
        mask_size = 3 * int(math.ceil(imgs.size(-1) / 16))
        self.src_mask = torch.ones(imgs.size(0), 1, mask_size).cuda()

        self.trg = trg.cuda()
        self.trg_y = trg_y.cuda()

        trg_mask = (trg != pad).unsqueeze(-2)
        size = trg.size(-1) # max_len=100
        attn_shape = (1, size, size)

        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0)

        trg_mask = trg_mask & subsequent_mask.type_as(trg_mask.data)
        #trg_mask [32,max_len,max_len] 下三角为1矩阵

        self.trg_mask = trg_mask.cuda()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name

    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

class resizeNormalize(object):

    def __init__(self, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        W, H = img.size
        ratio = W / float(H)
        imgW = int(ratio * self.imgH)
        img = img.resize((imgW, self.imgH), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=48, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels_y, labels, lengths = zip(*batch) #zipped=zip(a,b)：压成一个个元组  zip(*zipped)：解压成 a,b
        labels_y = torch.IntTensor(labels_y)
        labels = torch.IntTensor(labels)
        imgH = self.imgH

        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        #图片是按照（w,h）
        transform = resizeNormalize(imgH)
        images_list = []
        max_width = 0
        for img in images:
            temp = transform(img)
            if temp.size(2) > max_width: max_width = temp.size(2)
            images_list.append(transform(img))

        resize_imgs = []
        for img in images_list:
            m = nn.ZeroPad2d(padding=(0, max_width-img.size(2), 0, 0))
            img = m(img)
            resize_imgs.append(img)
        images = torch.cat([item.unsqueeze(0) for item in resize_imgs], 0)
        return images, labels_y, labels, lengths

if __name__ == '__main__':
    train_lm = lmdbDataset("D:/Document/DataSet/reg_dataset/NIPS2014")
    print(type(train_lm[0][0]))
    dataloader = DataLoader(train_lm, batch_size=2, num_workers=0,
                             shuffle=True, pin_memory=True, drop_last=True,
                             collate_fn=alignCollate(imgH=32, keep_ratio=False))
    dataiter = iter(dataloader)
    imgs,label_y,label,lens = next(dataiter)
    imgs=torch.tensor(imgs)
    label_y=torch.tensor(label_y)
    print(imgs.shape,label_y.shape)
    print(lens)


















