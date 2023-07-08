from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms

def create_dataloader(bs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR100('.', download=True, transform=transform)
    sample_image, _ = dataset[0]
    image_channels,image_height,image_width = sample_image.shape
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    return dataloader
