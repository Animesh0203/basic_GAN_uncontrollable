from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, image_height, image_width):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_channels * image_height * image_width),
            nn.ReLU()
        )
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, noise):
        batch_size = noise.shape[0]
        x = self.gen(noise)
        x = x.view(batch_size, self.image_channels, self.image_height, self.image_width)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_channels, image_height, image_width):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_channels * image_height * image_width, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, image):
        batch_size = image.shape[0]
        x = image.view(batch_size, -1)
        x = self.disc(x)
        return x
