from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms

def create_dataloader(bs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR100(root=root, transform=transform, download=True)
    sample_image, _ = dataset[0]
    image_channels, image_height, image_width = sample_image.shape

    return dataset, image_channels, image_height, image_width
