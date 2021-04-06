import torch
import torch.nn as nn
from torch.utils.data import Dataset
import lmdb
import six
from PIL import Image
import numpy as np
import labelsmap
import torchvision.transforms as transforms
import math
from logger import get_logger
import os.path as osp

labelTool = labelsmap.LabelTool()

class LmdbDataset(Dataset):
    def __init__(self, root, max_len = 100):
        super(LmdbDataset, self).__init__()
        self.env = lmdb.open(root)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.voc = labelTool.voc
        self.char2id = labelTool.char2id
        self.id2char = labelTool.id2char
        self.max_len = max_len
        self.log=open("/home/gmn/Transformer/logs/unseecharacter.txt","w")


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index += 1
        image_key = 'image-%09d' % index
        buf = six.BytesIO()
        image = self.txn.get(image_key.encode())
        buf.write(image)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')

        label_key = 'label-%09d' % index
        word = self.txn.get(label_key.encode()).decode()
        label = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate('<' + word):
            if c in self.voc:
                label[i] = self.char2id[c]
            else:
                label[i] = self.char2id["PADDING"]
                self.log.write(c)


        label_y = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate(word + '>'):
            if c in self.voc:
                label_y[i] = self.char2id[c]
            else:
                label[i] = self.char2id["PADDING"]
                self.log.write(c)

        return image, label_y, label, len(word)+1


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


class AlignCollate(object):
    def __init__(self, imgH=32, keep_ratio=False, min_ratio=1):
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

class Batch(object):
    def __init__(self, images, labels_y, labels, pad=0):
        self.src = images.cuda()

        # print("images.shape",images.shape)

        # print("labels_y.shape", labels_y.shape)
        # print(labels_y[0])

        mask_size = (int(math.ceil(images.size(-2) / 16))) * int(math.ceil(images.size(-1) / 16))
        self.src_mask = torch.ones(images.size(0), 1, mask_size).cuda()

        # print("src_mask.shape",self.src_mask.shape)
        # print(self.src_mask)

        self.trg = labels.cuda()
        trg_mask = (labels != pad).unsqueeze(-2)

        # print("trg_mask", trg_mask.shape)

        size = labels.size(-1) # max_len=100
        attn_shape = (1, size, size)

        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = (torch.from_numpy(subsequent_mask) == 0)
        # print("subsequent_mask ",subsequent_mask)

        # print("subsequent_mask",subsequent_mask.shape)

        trg_mask = trg_mask & subsequent_mask.type_as(trg_mask.data)

        # print("trg_mask",trg_mask.shape)

        self.trg_mask = trg_mask.cuda()
        self.trg_y = labels_y.cuda()


