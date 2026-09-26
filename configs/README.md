# 实验配置目录

本目录用于保存可复现的实验配置。当前 `ms3.py` 和 `ms3_val.py` 仍使用脚本内的 `EasyDict`，尚未实现外部配置加载，因此这里暂不创建会被误认为可直接执行的 YAML 配置。

后续接入配置系统时，建议按实验编号命名，例如：

```text
configs/
├── E0_baseline.yaml
├── E1_augmentation.yaml
└── E2_weight_decay.yaml
```

每份配置至少应覆盖：数据划分清单或版本、seed、模型、batch size、epochs、optimizer、learning rate、scheduler、augmentation、weight decay、dropout、class weight、输出目录和设备设置。

配置文件中不得写入本机绝对路径、账号、密钥或其他隐私信息。路径应相对项目根目录，或通过不提交 Git 的环境变量/本地配置提供。

