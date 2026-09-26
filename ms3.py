from easydict import EasyDict as edict
#glob模块主要用于查找符合特定规则的文件路径名，类似使用windows下的文件搜索
import glob
#os模块主要用于处理文件和目录
import os
 
import numpy as np
import matplotlib.pyplot as plt
 
import mindspore
#导入mindspore框架数据集
import mindspore.dataset as ds
#vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。
import mindspore.dataset.vision.c_transforms as CV
#c_transforms模块提供常用操作，包括OneHotOp和TypeCast
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore import context
#导入模块用于初始化截断正态分布
from mindspore.common.initializer import TruncatedNormal
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

# 设置MindSpore的执行模式和设备
context.set_context(device_target="CPU", mode=mindspore.GRAPH_MODE)

import random
 
seed = 42  # 设定随机种子
random.seed(seed)
np.random.seed(seed)
mindspore.set_seed(seed)

cfg = edict({
    'data_path': './flower_photos',
    'data_size':3670,
    'image_width': 100,  # 图片宽度
    'image_height': 100,  # 图片高度
    'batch_size': 32,
    'channel': 3,  # 图片通道数
    'num_class':5,  # 分类类别
    'weight_decay': 0.01,
    'lr':0.0001,  # 学习率
    'dropout_ratio': 0.5,
    'epoch_size': 10,  # 训练次数
    'per_print_times': 10,  # 每10步打印一次
    'sigma':0.01,
    'save_checkpoint_steps': 1000,  # 添加检查点保存步数
    'keep_checkpoint_max': 10,  # 设置最大保存的检查点数量
    'output_prefix': 'restrain_output_10',  # 保存的模型文件前缀
    'output_directory': './restrain_output_10'  # 保存的模型文件路径
})

#从目录中读取图像的源数据集。
de_dataset = ds.ImageFolderDataset(cfg.data_path,
                                   class_indexing={'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4})
#解码前将输入图像裁剪成任意大小和宽高比。
transform_img = CV.RandomCropDecodeResize([cfg.image_width,cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸
#转换输入图像；形状（H, W, C）为形状（C, H, W）。
hwc2chw_op = CV.HWC2CHW()
#转换为给定MindSpore数据类型的Tensor操作。
type_cast_op = C.TypeCast(mstype.float32)
#将操作中的每个操作应用到此数据集。
de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)
de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)
de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)
#划分训练集测试集
(de_train,de_test)=de_dataset.split([0.8,0.2])
#设置每个批处理的行数
#drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
#如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。
de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
#重复此数据集计数次数。
de_train=de_train.repeat(cfg.epoch_size)
de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
de_test=de_test.repeat(cfg.epoch_size)
print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size)#get_dataset_size()获取批处理的大小。
print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)


# 定义CNN图像识别网络
class ResidualBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, pad_mode='same', weight_init=TruncatedNormal(0.02))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, pad_mode='same', weight_init=TruncatedNormal(0.02))
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, pad_mode='same', weight_init=TruncatedNormal(0.02)),
                nn.BatchNorm2d(out_channels)
            ])
 
    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
        return out
 
class ResNet18(nn.Cell):
    def __init__(self, num_class=5):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='same', weight_init=TruncatedNormal(0.02))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
 
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512, num_class, weight_init=TruncatedNormal(0.02))
 
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.SequentialCell(layers)
 
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
 
net = ResNet18(num_class=cfg.num_class)

#计算softmax交叉熵。
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
#opt
# 从所有可训练参数中筛选出全连接层(fc)的权重参数
fc_weight_params = list(filter(lambda x: 'fc' in x.name and 'weight' in x.name, net.trainable_params()))
# 创建全连接层权重参数的ID集合，用于快速查找
fc_weight_param_ids = {id(param) for param in fc_weight_params}
# 获取除全连接层权重外的所有其他参数
else_params = [param for param in net.trainable_params() if id(param) not in fc_weight_param_ids]
# 全连接层权重衰减，其他不衰减，避免全连接层过拟合（全连接层参数量大）
group_params = [{'params': fc_weight_params, 'weight_decay': cfg.weight_decay},
                {'params': else_params}]
#设置Adam优化器
net_opt = nn.Adam(group_params, learning_rate=cfg.lr, weight_decay=0.0)
 
loss_list = []
class CustomLossMonitor(LossMonitor):
    # 在每个训练epoch结束时自动调用，重写
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss_list.append(cb_params.net_outputs.asnumpy())  # 记录每个 epoch 的损失
        super().epoch_end(run_context)  # 调用原 LossMonitor 的方法

model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})
loss_cb = CustomLossMonitor(per_print_times=de_train.get_dataset_size())
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=cfg.output_directory, config=config_ck)
print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)
 
plt.figure()
plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig(r'E:\CodeFile\pythonProject\pythonProject\flowers\train_loss_curve.png')  # 保存图片
plt.close()  # 关闭图像，不显示

# 使用测试集评估模型，打印总体准确率
print("============== Starting Evaluation ==============")
metric = model.eval(de_test,dataset_sink_mode=False)
print(metric)