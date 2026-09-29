"""Loss cells shared by experiments that need explicit sample weighting."""

import numpy as np

import mindspore
from mindspore import nn, ops
from mindspore.common import dtype as mstype


class WeightedSoftmaxCrossEntropy(nn.Cell):
    """Mean of sparse per-sample CE multiplied by the target class weight."""

    def __init__(self, class_weights):
        super().__init__()
        class_weights = np.asarray(class_weights, dtype=np.float32)
        if class_weights.ndim != 1 or class_weights.size == 0:
            raise ValueError("class_weights must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(class_weights)) or np.any(class_weights <= 0):
            raise ValueError("Class weights must be finite and strictly positive.")
        self.class_weights = mindspore.Tensor(class_weights, mstype.float32)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction='none'
        )
        self.reduce_mean = ops.ReduceMean()

    def construct(self, logits, labels):
        per_sample_loss = self.cross_entropy(logits, labels)
        sample_weights = ops.gather(self.class_weights, labels, 0)
        return self.reduce_mean(per_sample_loss * sample_weights)
