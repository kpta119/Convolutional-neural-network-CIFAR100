import torch
from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from cifar_dataloader import classes

def eval(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_targets = []
    all_predictions = []
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data in loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for label, prediction in zip(targets, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    the_best_predicted, the_worst_predicted =  n_best_and_worst_predicted_classes(3, correct_pred, total_pred)
    avg_loss = total_loss / len(loader)
    metrics = count_metrics(all_targets, all_predictions, correct, total)

    _, worst_precision_class, _, _ = display_best_and_worst_precision_recall(all_targets, all_predictions)
    best_classes = [classname for classname, _ in the_best_predicted]
    worst_classes = [classname for classname, _ in the_worst_predicted]
    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(len(classes))))
    plot_confusion_matrix(cm, best_classes, 'best_classes_confusion_matrix')
    plot_confusion_matrix(cm, worst_classes, 'worst_classes_confusion_matrix')
    plot_confusion_matrix_for_FP(cm, [worst_precision_class], 'worst_precision_class')

    return avg_loss, metrics


def n_best_and_worst_predicted_classes(n, correct_pred, total_pred):
    sorted_classes = sorted(correct_pred.items(), key=lambda item: (100 * item[1] / total_pred[item[0]]), reverse=True)
    the_best_predicted = sorted_classes[:n]
    the_worst_predicted = sorted_classes[-n:]

    for classname, correct_count in the_best_predicted:
        print(f"{classname:.10s}: {100* correct_count/total_pred[classname]}")

    for classname, correct_count in the_worst_predicted:
        print(f"{classname:.10s}: {100*correct_count/total_pred[classname]}")

    return the_best_predicted, the_worst_predicted


def count_metrics(all_targets, all_predictions, correct, total):
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=1)
    accuracy = correct / total
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics


def plot_confusion_matrix(cm, target_classes, filename, save=False):
        i=1
        for target_class in target_classes:
            target_index = classes.index(target_class)
            class_cm = cm[target_index, :]
            top_indices = class_cm.argsort()[-15:][::-1]
            top_classes = [classes[i] for i in top_indices]
            top_values = class_cm[top_indices]
            plt.figure(figsize=(10, 7))
            sns.heatmap(top_values.reshape(1, -1), annot=True, fmt='d', xticklabels=top_classes, yticklabels=[target_class], cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {target_class}')
            if save:
                plt.savefig(f'{filename}_{i}.png')
                i+=1
            plt.show()
            plt.close()


def plot_confusion_matrix_for_FP(cm, target_classes, filename, save=False):
        i=1
        for target_class in target_classes:
            target_index = classes.index(target_class)
            class_cm = cm[: ,target_index]
            top_indices = class_cm.argsort()[-5:][::-1]
            top_classes = [classes[i] for i in top_indices]
            top_values = class_cm[top_indices]
            plt.figure(figsize=(10,7))
            sns.heatmap(top_values.reshape(-1,1), annot=True, fmt='d', yticklabels=top_classes, xticklabels=[target_class], cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {target_class}')
            if save:
                plt.savefig(f'{filename}_{i}.png')
                i+=1
            plt.show()
            plt.close()


def display_best_and_worst_precision_recall(all_targets, all_predictions):
    class_precision = precision_score(all_targets, all_predictions, average=None, zero_division=1)
    class_recall = recall_score(all_targets, all_predictions, average=None, zero_division=1)

    best_precision_class = classes[class_precision.argmax()]
    worst_precision_class = classes[class_precision.argmin()]
    best_recall_class = classes[class_recall.argmax()]
    worst_recall_class = classes[class_recall.argmin()]

    print(f"\nKlasa z największym precision: {best_precision_class} ({class_precision.max():.2f})")
    print(f"Klasa z najmniejszą precision: {worst_precision_class} ({class_precision.min():.2f})")
    print(f"Klasa z największym recall: {best_recall_class} ({class_recall.max():.2f})")
    print(f"Klasa z najmniejszym recall: {worst_recall_class} ({class_recall.min():.2f})")

    return best_precision_class, worst_precision_class, best_recall_class, worst_recall_class