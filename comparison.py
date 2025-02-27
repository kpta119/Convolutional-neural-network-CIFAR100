import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from cifar_dataloader import testloader as c_testloader
from backbone_dataloader import testloader as b_testloader
from backbone_dataloader import trainloader as b_trainloader
from backbone_dataloader import val_loader as b_val_loader
from eval import eval
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    path: str= "conv5_3v2.pth"  #path to saved model
    resnet_path: str = "resnet18v1.pth"  #path to saved resnet model
    criterion: nn.Module = nn.CrossEntropyLoss()

def main(config: Config):
    my_conv_model = torch.load(config.path)
    my_conv_model.to(device)
    my_conv_model.eval()

    resnet18 = models.resnet18(weights=None)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )
    resnet18.load_state_dict(torch.load(config.resnet_path))
    resnet18.to(device)

    models_dict = {
        "My model": [my_conv_model, c_testloader],
        "ResNet-18": [resnet18, b_testloader],
    }

    accuracies = {}
    for name, net in models_dict.items():
        loss, metrics = eval(net[0], config.criterion, net[1], device)
        acc = metrics["accuracy"]
        acc = acc * 100
        accuracies[name] = acc
        print(f"{name}: {acc:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values(),
            color=['blue', 'red'])
    plt.xticks(rotation=15, ha='right')
    plt.ylabel("Accuracy [%]")
    plt.title("Model accuracy comparison on CIFAR-100 (test set)")
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = Config()
    main(config)

