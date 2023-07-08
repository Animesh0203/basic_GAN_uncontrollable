import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataset import create_dataloader
from models import Generator, Discriminator
from utils import show_images
from loss import calc_gen_loss, calc_disc_loss

# setup of the main parameters and hyperparameters
epochs = 50000
cur_step = 0
info_step = 300
mean_gen_loss = 0
mean_disc_loss = 0

z_dim = 64
lr = 0.00001
bs = 128
device = 'cuda'

# Create dataloader
dataloader, image_channels, image_height, image_width = create_dataloader("dataset/")

# Create generator and discriminator
gen = Generator(z_dim)
disc = Discriminator()

# Move models to device
gen.to(device)
disc.to(device)

# Define optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Define loss function
loss_func = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        # Move real data to device
        real = real.to(device)

        # Discriminator training
        disc_opt.zero_grad()
        disc_loss = calc_disc_loss(loss_func, gen, disc, real, z_dim, device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Generator training
        gen_opt.zero_grad()
        gen_loss = calc_gen_loss(loss_func, gen, disc, bs, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        # Visualization & stats
        mean_disc_loss += disc_loss.item() / info_step
        mean_gen_loss += gen_loss.item() / info_step

        if cur_step % info_step == 0 and cur_step > 0:
            # Generate fake images
            fake_noise = torch.randn(bs, z_dim).to(device)
            fake = gen(fake_noise)

            # Show generated and real images
            show_images(fake, real)

            print(f"{epoch}: step {cur_step} / Gen loss: {mean_gen_loss} / disc_loss: {mean_disc_loss}")
            mean_gen_loss, mean_disc_loss = 0, 0
        cur_step += 1
