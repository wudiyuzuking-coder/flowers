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

from evaluation import evaluate_dataset, save_test_evaluation, save_training_curves
from evaluation import save_confusion_matrix
from split_manifest import CLASS_INDEXING, ManifestImageSource
from split_manifest import load_and_validate_split_manifests

# 设置MindSpore的执行模式和设备
context.set_context(device_target="GPU", mode=mindspore.GRAPH_MODE)

import random
 
seed = 42  # 设定随机种子
random.seed(seed)
np.random.seed(seed)
mindspore.set_seed(seed)
ds.config.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the E0-E3 flower experiments.")
    parser.add_argument(
        "--experiment-id",
        choices=("E0", "E1", "E2", "E3"),
        default="E0",
        help=(
            "E0 is the baseline; E1 adds conservative augmentation; "
            "E2 keeps E1 and extends weight decay to backbone weights; "
            "E3 keeps E2 and adds cosine learning-rate decay."
        ),
    )
    return parser.parse_args()


args = parse_args()

cfg = edict({
    'experiment_id': args.experiment_id,
    'use_extra_augmentation': args.experiment_id in ('E1', 'E2', 'E3'),
    'use_backbone_weight_decay': args.experiment_id in ('E2', 'E3'),
    'use_lr_scheduler': args.experiment_id == 'E3',
    'data_path': './flower_photos',
    'splits_path': './splits',
    'data_size':3670,
    'image_width': 100,  # 图片宽度
    'image_height': 100,  # 图片高度
    'batch_size': 32,
    'channel': 3,  # 图片通道数
    'num_class':5,  # 分类类别
    'weight_decay': 0.01,
    'backbone_weight_decay': 0.0001,
    'lr':0.0001,  # 学习率
    'min_lr': 0.000001,
    'dropout_ratio': 0.5,
    'epoch_size': 10,  # 训练次数
    'per_print_times': 10,  # 每10步打印一次
    'sigma':0.01,
    'save_checkpoint_steps': 1000,  # 添加检查点保存步数
    'keep_checkpoint_max': 10,  # 设置最大保存的检查点数量
    'output_prefix': f'{args.experiment_id}_best',  # 最佳验证集模型文件前缀
    'output_directory': f'./checkpoints/{args.experiment_id}'  # 保存的模型文件路径
})

def create_datasets(config):
    """Build every dataset from the same validated, fixed split manifests."""
    split_entries, split_info = load_and_validate_split_manifests(
        config.data_path, config.splits_path
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

    train_source = ManifestImageSource(
        config.data_path, split_entries['train'], CLASS_INDEXING
    )
    val_source = ManifestImageSource(
        config.data_path, split_entries['val'], CLASS_INDEXING
    )
    test_source = ManifestImageSource(
        config.data_path, split_entries['test'], CLASS_INDEXING
    )
    train_dataset = ds.GeneratorDataset(
        train_source, column_names=['image', 'label'], shuffle=False
    )
    train_eval_dataset = ds.GeneratorDataset(
        train_source, column_names=['image', 'label'], shuffle=False
    )
    val_dataset = ds.GeneratorDataset(
        val_source, column_names=['image', 'label'], shuffle=False
    )
    test_dataset = ds.GeneratorDataset(
        test_source, column_names=['image', 'label'], shuffle=False
    )

    train_dataset = train_dataset.map(
        input_columns="image", operations=train_operations, num_parallel_workers=8
    )
    train_dataset = train_dataset.shuffle(buffer_size=len(train_source))
    train_eval_dataset = train_eval_dataset.map(
        input_columns="image", operations=eval_operations, num_parallel_workers=8
    )
    val_dataset = val_dataset.map(
        input_columns="image", operations=eval_operations, num_parallel_workers=8
    )
    test_dataset = test_dataset.map(
        input_columns="image", operations=eval_operations, num_parallel_workers=8
    )

    # model.train controls epoch repetition. Validation and test are single-pass.
    train_dataset = train_dataset.batch(config.batch_size, drop_remainder=True)
    train_eval_dataset = train_eval_dataset.batch(
        config.batch_size, drop_remainder=False
    )
    val_dataset = val_dataset.batch(config.batch_size, drop_remainder=False)
    test_dataset = test_dataset.batch(config.batch_size, drop_remainder=False)
    return train_dataset, train_eval_dataset, val_dataset, test_dataset, split_info


de_train, de_train_eval, de_val, de_test, split_info = create_datasets(cfg)
print('实验编号：', cfg.experiment_id)
print('启用额外数据增强：', cfg.use_extra_augmentation)
print('启用 Backbone Weight Decay：', cfg.use_backbone_weight_decay)
print('启用 Learning Rate Scheduler：', cfg.use_lr_scheduler)
print('训练批次数：', de_train.get_dataset_size())
print('验证批次数：', de_val.get_dataset_size())
print('测试批次数：', de_test.get_dataset_size())
print('固定 split 样本数：', split_info['split_counts'])
print('固定 split manifest SHA-256：', split_info['manifest_sha256'])


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


def create_learning_rate(config, steps_per_epoch):
    """Return a fixed LR or an exact per-step cosine schedule Tensor."""
    if not config.use_lr_scheduler:
        return config.lr, None

    total_steps = int(steps_per_epoch * config.epoch_size)
    if total_steps < 2:
        raise ValueError("Cosine learning-rate decay requires at least two training steps.")

    progress = np.linspace(0.0, 1.0, total_steps, dtype=np.float64)
    learning_rate_values = config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1.0 + np.cos(np.pi * progress)
    )
    learning_rate_values = learning_rate_values.astype(np.float32)
    learning_rate_tensor = mindspore.Tensor(learning_rate_values, mstype.float32)
    return learning_rate_tensor, learning_rate_values


def save_learning_rate_curve(learning_rate_values, experiment_id):
    """Save the configured schedule values; this does not change training."""
    curve_directory = './results/curves'
    os.makedirs(curve_directory, exist_ok=True)
    curve_path = os.path.join(curve_directory, f'{experiment_id}_lr_curve.png')
    plt.figure()
    plt.plot(
        np.arange(1, len(learning_rate_values) + 1),
        learning_rate_values,
        linestyle='-',
    )
    plt.xlabel('Optimizer Step')
    plt.ylabel('Learning Rate')
    plt.title(f'{experiment_id} Configured Cosine Learning Rate')
    plt.grid(True)
    plt.savefig(curve_path)
    plt.close()
    return curve_path


steps_per_epoch = de_train.get_dataset_size()
optimizer_learning_rate, configured_lr_values = create_learning_rate(
    cfg, steps_per_epoch
)
if configured_lr_values is None:
    print(f"Learning rate: fixed at {cfg.lr}")
else:
    lr_curve_path = save_learning_rate_curve(configured_lr_values, cfg.experiment_id)
    print("Learning rate scheduler: per-step cosine decay")
    print(f"Initial learning rate: {configured_lr_values[0]:.8f}")
    print(f"Final learning rate: {configured_lr_values[-1]:.8f}")
    print(f"Total decay steps: {len(configured_lr_values)}")
    print(f"Configured learning rate curve: {lr_curve_path}")
#设置Adam优化器
net_opt = nn.Adam(group_params, learning_rate=optimizer_learning_rate, weight_decay=0.0)
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": nn.Accuracy()})
loss_cb = LossMonitor(per_print_times=de_train.get_dataset_size())


class BestValidationCheckpoint(Callback):
    """Record deterministic train/val metrics and select by validation accuracy."""
    def __init__(
        self,
        model_to_eval,
        network,
        train_eval_dataset,
        val_dataset,
        checkpoint_path,
    ):
        super().__init__()
        self.model_to_eval = model_to_eval
        self.network = network
        self.train_eval_dataset = train_eval_dataset
        self.val_dataset = val_dataset
        self.checkpoint_path = checkpoint_path
        self.best_accuracy = -1.0
        self.best_epoch = None
        self.best_epoch_train_accuracy = None
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = int(cb_params.cur_epoch_num)
        selection_metrics = self.model_to_eval.eval(
            self.val_dataset, dataset_sink_mode=False
        )
        accuracy = float(selection_metrics['Accuracy'])
        train_metrics, _, _ = evaluate_dataset(
            self.model_to_eval, self.train_eval_dataset
        )
        val_metrics, _, _ = evaluate_dataset(self.model_to_eval, self.val_dataset)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(accuracy)
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.6f}, "
            f"train_accuracy={train_metrics['accuracy']:.6f}, "
            f"val_loss={val_metrics['loss']:.6f}, val_accuracy={accuracy:.6f}"
        )
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            self.best_epoch_train_accuracy = float(train_metrics['accuracy'])
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            save_checkpoint(self.network, self.checkpoint_path)
            print(f"Saved new best validation checkpoint: {self.checkpoint_path}")


best_checkpoint_path = os.path.join(
    cfg.output_directory, f"{cfg.output_prefix}.ckpt"
)
validation_cb = BestValidationCheckpoint(
    model, net, de_train_eval, de_val, best_checkpoint_path
)
print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, validation_cb], dataset_sink_mode=True)
training_curve_path = save_training_curves(validation_cb.history, cfg.experiment_id)
print('Training/validation curves:', training_curve_path)

# 测试集只评估由 validation accuracy 选出的最佳 checkpoint。
print("============== Starting Evaluation ==============")
load_checkpoint(best_checkpoint_path, net=net)
test_metrics, _, _ = evaluate_dataset(model, de_test)
confusion_matrix_path = save_confusion_matrix(
    test_metrics['confusion_matrix'],
    cfg.experiment_id,
    './results/confusion_matrix',
)
evaluation_output, evaluation_path = save_test_evaluation(
    test_metrics,
    cfg.experiment_id,
    validation_cb.best_epoch,
    validation_cb.best_accuracy,
    validation_cb.best_epoch_train_accuracy,
    best_checkpoint_path,
)
print('Best validation epoch:', validation_cb.best_epoch)
print('Best validation accuracy:', validation_cb.best_accuracy)
print('Best epoch train accuracy:', validation_cb.best_epoch_train_accuracy)
print('Test evaluation:', evaluation_output)
print('Evaluation JSON:', evaluation_path)
print('Confusion matrix:', confusion_matrix_path)
