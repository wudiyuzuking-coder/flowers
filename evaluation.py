"""Shared, deterministic evaluation and plotting for all flower experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')


def _as_numpy(value):
    return value.asnumpy() if hasattr(value, 'asnumpy') else np.asarray(value)


def collect_predictions(model, dataset):
    """Collect every sample once and compute mean sparse cross-entropy from logits."""
    true_batches = []
    prediction_batches = []
    loss_sum = 0.0
    sample_count = 0
    for batch in dataset.create_dict_iterator(num_epochs=1):
        labels = _as_numpy(batch['label']).astype(np.int64).reshape(-1)
        logits = _as_numpy(model.predict(batch['image'])).astype(np.float64)
        if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
            raise ValueError("Logits and labels have incompatible shapes.")
        predictions = np.argmax(logits, axis=1)
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted_logits).sum(axis=1))
        batch_losses = log_sum_exp - shifted_logits[np.arange(labels.size), labels]
        loss_sum += float(batch_losses.sum())
        sample_count += int(labels.size)
        true_batches.append(labels)
        prediction_batches.append(predictions)
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return (
        np.concatenate(true_batches),
        np.concatenate(prediction_batches),
        loss_sum / sample_count,
    )


def compute_classification_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Compute fixed-label metrics with zero_division=0 semantics."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal shape.")
    class_count = len(class_names)
    if (
        np.any(y_true < 0)
        or np.any(y_true >= class_count)
        or np.any(y_pred < 0)
        or np.any(y_pred >= class_count)
    ):
        raise ValueError("Labels must be within the configured class range.")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    worst_index = int(np.argmin(recall))
    per_class = {
        str(index): {
            'name': class_name,
            'precision': float(precision[index]),
            'recall': float(recall[index]),
            'f1': float(f1[index]),
            'support': int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        'accuracy': float(true_positive.sum() / y_true.size),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'worst_class_recall': float(recall[worst_index]),
        'worst_class_name': class_names[worst_index],
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
    }


def evaluate_dataset(model, dataset):
    y_true, y_pred, mean_loss = collect_predictions(model, dataset)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics['loss'] = float(mean_loss)
    return metrics, y_true, y_pred


def save_confusion_matrix(confusion, experiment_id, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f'{experiment_id}_confusion_matrix.png'
    confusion = np.asarray(confusion, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion, cmap='Blues')
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel='Predicted',
        ylabel='True',
        title=f'{experiment_id} Confusion Matrix',
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha='right')
    threshold = confusion.max() / 2.0 if confusion.size else 0.0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha='center',
                va='center',
                color='white' if confusion[row, column] > threshold else 'black',
            )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_training_curves(history, experiment_id, output_directory='results/curves'):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f'{experiment_id}_training_curves.png'
    epochs = np.arange(1, len(history['train_loss']) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history['train_loss'], marker='o', label='Train')
    axes[0].plot(epochs, history['val_loss'], marker='o', label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-Entropy Loss', title='Loss')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(epochs, history['train_accuracy'], marker='o', label='Train')
    axes[1].plot(epochs, history['val_accuracy'], marker='o', label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    axes[1].grid(True)
    axes[1].legend()
    figure.suptitle(f'{experiment_id} Training and Validation Curves')
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_test_evaluation(
    metrics,
    experiment_id,
    best_epoch,
    best_val_accuracy,
    best_epoch_train_accuracy,
    checkpoint_path,
    output_directory='results/evaluation',
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    train_test_gap = float(best_epoch_train_accuracy - metrics['accuracy'])
    output = {
        'experiment_id': experiment_id,
        'accuracy': metrics['accuracy'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
        'worst_class_recall': metrics['worst_class_recall'],
        'worst_class_name': metrics['worst_class_name'],
        'train_test_gap': train_test_gap,
        'best_epoch': int(best_epoch),
        'best_val_accuracy': float(best_val_accuracy),
        'best_epoch_train_accuracy': float(best_epoch_train_accuracy),
        'checkpoint': str(checkpoint_path),
        'per_class': metrics['per_class'],
        'confusion_matrix': metrics['confusion_matrix'],
    }
    output_path = output_directory / f'{experiment_id}_metrics.json'
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return output, output_path
