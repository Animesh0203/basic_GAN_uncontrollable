from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms

def create_dataloader(bs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR100('.', download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return dataloader
