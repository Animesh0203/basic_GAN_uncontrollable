import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_images(fake, real):
    fake_grid = make_grid(fake.detach().cpu(), nrow=4)
    real_grid = make_grid(real.detach().cpu(), nrow=4)

    plt.imshow(fake_grid.permute(1, 2, 0))
    plt.show()

    plt.imshow(real_grid.permute(1, 2, 0))
    plt.show()
