from matplotlib import pyplot as plt


def imshow(img):
    """Function to show an image."""
    img = img / 2 + 0.5  # unnormalize if normalization was applied during pre-processing
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_latent_space_helper(model, test_loader, metaData=None, exp_dir=None, device=None):
    model.eval()
