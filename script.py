import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from train_epoch import train_epoch
import torchvision.models as models
from model_conv5 import Conv
from eval import eval
from cifar_dataloader import testloader, trainloader
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Type


device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    save: bool = False
    load: bool = True
    load_path: str = 'model_conv5.pth'
    save_path: str = 'conv5_3v8.pth'
    model_class: Type[nn.Module] = Conv
    epochs: int = 40
    lr: float = 1e-3
    criterion: nn.Module = nn.CrossEntropyLoss()


def main(config: Config):
    if config.load:
        model = torch.load(config.load_path)
    else:
        model = config.model_class()
        epoch_losses = []
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
        train_losses = []
        test_losses = []
        for epoch in range(config.epochs):
            epoch_loss, epoch_accuracy = train_epoch(model, config.criterion, optimizer, trainloader, device)
            epoch_losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            train_losses.append(epoch_loss)
            test_loss, _ = eval(model, config.criterion, testloader, device)
            test_losses.append(test_loss)
        avg_loss, metrics = eval(model, config.criterion, testloader, device)
        print(f"Loss: {avg_loss:.4f}, Test Accuracy: {metrics["accuracy"]*100:.2f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1_score']:.4f}" )
        if config.save:
            torch.save(model, config.save_path)
        plt.plot(range(1, config.epochs + 1), train_losses, label="Train")
        plt.plot(range(1, config.epochs + 1), test_losses, label="Test")
        plt.xlabel('Epoch', fontweight="bold")
        plt.ylabel('Loss', fontweight="bold")
        plt.title('Loss for epoch', fontweight="bold")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    config = Config()
    main(config)
