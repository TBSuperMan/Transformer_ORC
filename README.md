# Transformer_ORC

复现文章(ICDAR 2019)NRTR A No-Recurrence Sequence-to-Sequence Model For Scene Text

训练数据集：Synth(NIPS2014)  8百万张图片
测试数据集：IIIT5K_3000      3千张图片

模型：Resnet101+Transformer

掩码：
src_mask [batch,1,W/16*H/16] 全为1的矩阵     目的：让selfattention不对填充进行关注
代码中max_len设置为100
trg_mask [batch,max_len,max_len] 下三角为1矩阵   目的：防止decoder时看到未来的信息
