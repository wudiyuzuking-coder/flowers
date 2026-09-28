from easydict import EasyDict as edict
import argparse
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
from mindspore.train.callback import Callback, LossMonitor
from mindspore.train.serialization import load_checkpoint, save_checkpoint

# 设置MindSpore的执行模式和设备
context.set_context(device_target="GPU", mode=mindspore.GRAPH_MODE)

import random
 
seed = 42  # 设定随机种子
random.seed(seed)
np.random.seed(seed)
mindspore.set_seed(seed)
ds.config.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the E0, E1, or E2 flower experiment.")
    parser.add_argument(
        "--experiment-id",
        choices=("E0", "E1", "E2"),
        default="E0",
        help=(
            "E0 is the baseline; E1 adds conservative augmentation; "
            "E2 keeps E1 and extends weight decay to backbone weights."
        ),
    )
    return parser.parse_args()


args = parse_args()

cfg = edict({
    'experiment_id': args.experiment_id,
    'use_extra_augmentation': args.experiment_id in ('E1', 'E2'),
    'use_backbone_weight_decay': args.experiment_id == 'E2',
    'data_path': './flower_photos',
    'data_size':3670,
    'image_width': 100,  # 图片宽度
    'image_height': 100,  # 图片高度
    'batch_size': 32,
    'channel': 3,  # 图片通道数
    'num_class':5,  # 分类类别
    'weight_decay': 0.01,
    'backbone_weight_decay': 0.0001,
    'lr':0.0001,  # 学习率
    'dropout_ratio': 0.5,
    'epoch_size': 10,  # 训练次数
    'per_print_times': 10,  # 每10步打印一次
    'sigma':0.01,
    'save_checkpoint_steps': 1000,  # 添加检查点保存步数
    'keep_checkpoint_max': 10,  # 设置最大保存的检查点数量
    'output_prefix': f'{args.experiment_id}_best',  # 最佳验证集模型文件前缀
    'output_directory': f'./checkpoints/{args.experiment_id}'  # 保存的模型文件路径
})

CLASS_INDEXING = {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


def create_datasets(config):
    """Split raw samples first, then attach split-specific preprocessing."""
    raw_dataset = ds.ImageFolderDataset(
        config.data_path,
        class_indexing=CLASS_INDEXING,
        shuffle=False,
    )

    # The fixed dataset seed makes this split repeatable for an unchanged dataset and
    # MindSpore version. Formal runs still need a checked-in, stratified split manifest.
    train_dataset, val_dataset, test_dataset = raw_dataset.split(
        [0.7, 0.1, 0.2], randomize=True
    )

    train_operations = [
        CV.RandomCropDecodeResize(
            [config.image_width, config.image_height],
            scale=(0.08, 1.0),
            ratio=(0.75, 1.333),
        )
    ]
    if config.use_extra_augmentation:
        train_operations.extend([
            CV.RandomHorizontalFlip(prob=0.5),
            CV.RandomRotation(degrees=10),
            CV.RandomColorAdjust(
                brightness=(0.9, 1.1),
                contrast=(0.9, 1.1),
                saturation=(0.9, 1.1),
                hue=(-0.05, 0.05),
            ),
        ])
    train_operations.extend([CV.HWC2CHW(), C.TypeCast(mstype.float32)])

    eval_operations = [
        CV.Decode(),
        CV.Resize([config.image_width, config.image_height]),
        CV.HWC2CHW(),
        C.TypeCast(mstype.float32),
    ]

    train_dataset = train_dataset.map(
        input_columns="image", operations=train_operations, num_parallel_workers=8
    )
    train_dataset = train_dataset.shuffle(buffer_size=config.data_size)
    val_dataset = val_dataset.map(
        input_columns="image", operations=eval_operations, num_parallel_workers=8
    )
    test_dataset = test_dataset.map(
        input_columns="image", operations=eval_operations, num_parallel_workers=8
    )

    # model.train controls epoch repetition. Validation and test are single-pass.
    train_dataset = train_dataset.batch(config.batch_size, drop_remainder=True)
    val_dataset = val_dataset.batch(config.batch_size, drop_remainder=False)
    test_dataset = test_dataset.batch(config.batch_size, drop_remainder=False)
    return train_dataset, val_dataset, test_dataset


de_train, de_val, de_test = create_datasets(cfg)
print('实验编号：', cfg.experiment_id)
print('启用额外数据增强：', cfg.use_extra_augmentation)
print('启用 Backbone Weight Decay：', cfg.use_backbone_weight_decay)
print('训练批次数：', de_train.get_dataset_size())
print('验证批次数：', de_val.get_dataset_size())
print('测试批次数：', de_test.get_dataset_size())


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
# opt
def create_optimizer_param_groups(network, config):
    """Build explicit decay groups without relying on parameter-name patterns."""
    trainable_params = list(network.trainable_params())
    fc_weight_id = id(network.fc.weight)
    fc_weight_params = [param for param in trainable_params if id(param) == fc_weight_id]
    if len(fc_weight_params) != 1:
        raise ValueError("Expected exactly one trainable FC weight parameter.")

    if not config.use_backbone_weight_decay:
        # Preserve the E0/E1 grouping: only fc.weight has weight decay.
        other_params = [param for param in trainable_params if id(param) != fc_weight_id]
        return [
            {'params': fc_weight_params, 'weight_decay': config.weight_decay},
            {'params': other_params, 'weight_decay': 0.0},
        ]

    # Select weights by their owning Cell type, not by version-dependent names.
    # The current backbone contains Conv2d layers; the Dense check keeps the rule
    # explicit if a non-FC Dense layer is later introduced.
    decay_weight_ids = set()
    for _, cell in network.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)) and hasattr(cell, 'weight'):
            if id(cell.weight) != fc_weight_id:
                decay_weight_ids.add(id(cell.weight))

    backbone_weight_params = [
        param for param in trainable_params if id(param) in decay_weight_ids
    ]
    no_decay_params = [
        param for param in trainable_params
        if id(param) != fc_weight_id and id(param) not in decay_weight_ids
    ]

    grouped_ids = [
        id(param)
        for group in (fc_weight_params, backbone_weight_params, no_decay_params)
        for param in group
    ]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {
        id(param) for param in trainable_params
    }:
        raise ValueError("Optimizer parameter groups must cover each trainable parameter once.")
    if not backbone_weight_params:
        raise ValueError("E2 requires at least one backbone Conv2d/Dense weight parameter.")

    return [
        {'params': fc_weight_params, 'weight_decay': config.weight_decay},
        {
            'params': backbone_weight_params,
            'weight_decay': config.backbone_weight_decay,
        },
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


group_params = create_optimizer_param_groups(net, cfg)
for group_index, group in enumerate(group_params):
    print(
        f"Optimizer group {group_index}: weight_decay={group['weight_decay']}, "
        f"parameters={[param.name for param in group['params']]}"
    )
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


class BestValidationCheckpoint(Callback):
    """Select and save the model using validation accuracy only."""
    def __init__(self, model_to_eval, network, val_dataset, checkpoint_path):
        super().__init__()
        self.model_to_eval = model_to_eval
        self.network = network
        self.val_dataset = val_dataset
        self.checkpoint_path = checkpoint_path
        self.best_accuracy = -1.0
        self.best_epoch = None

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        metrics = self.model_to_eval.eval(self.val_dataset, dataset_sink_mode=False)
        accuracy = float(metrics['Accuracy'])
        epoch = int(cb_params.cur_epoch_num)
        print(f"Validation epoch {epoch}: Accuracy={accuracy:.6f}")
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            save_checkpoint(self.network, self.checkpoint_path)
            print(f"Saved new best validation checkpoint: {self.checkpoint_path}")


best_checkpoint_path = os.path.join(
    cfg.output_directory, f"{cfg.output_prefix}.ckpt"
)
validation_cb = BestValidationCheckpoint(model, net, de_val, best_checkpoint_path)
print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, validation_cb], dataset_sink_mode=True)
 
plt.figure()
plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
curve_directory = os.path.join('./flower_savefig', cfg.experiment_id)
os.makedirs(curve_directory, exist_ok=True)
plt.savefig(os.path.join(curve_directory, 'train_loss_curve.png'))
plt.close()  # 关闭图像，不显示

# 测试集只评估由 validation accuracy 选出的最佳 checkpoint。
print("============== Starting Evaluation ==============")
load_checkpoint(best_checkpoint_path, net=net)
metric = model.eval(de_test,dataset_sink_mode=False)
print('Best validation epoch:', validation_cb.best_epoch)
print('Best validation accuracy:', validation_cb.best_accuracy)
print(metric)
