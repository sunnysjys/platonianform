from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse


class LatentSpacePlotter:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        dataset_zip = np.load(
            '../dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
        self.metadata = dataset_zip['metadata'][()]
        self.latents_bases = np.concatenate((self.metadata["latents_sizes"][::-1].cumprod()[::-1][1:],
                                             np.array([1,])))

    def _compute_q_zCx_single(self, dataloader, index):
        """Compute q(z|x) for a single data point x at a specified index."""
        # ... [rest of your existing method here]
        # Assuming the second element is the target/label
        x, _ = dataloader.dataset[index]
        # Add batch dimension and send to device
        x = x.unsqueeze(0).to(self.device)
        print("breakpoint 5")
        with torch.no_grad():
            # Get the parameters of q(z|x) for the specified data point
            mean, log_var = self.model.encoder(x)
            params_zCx = (mean, log_var)

            # Sample z from q(z|x)
            sample_zCx = self.model.reparameterize(mean, log_var)

            output = self.model.decoder(sample_zCx)

        return sample_zCx, params_zCx, output, mean, log_var

    def imshow(self, img):
        """Function to show an image."""
        # ... [rest of your existing method here]
        img = img / 2 + 0.5  # unnormalize if normalization was applied during pre-processing
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def get_user_input(self, prompt, valid_choices):
        while True:
            try:
                value = int(input(prompt))
                if value in valid_choices:
                    return value
                else:
                    print("Invalid input. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def ask_user_input(self):
        print("Choose your shape and position:")

        shape = self.get_user_input(
            "Choose a shape (0 for square, 1 for ellipse, 2 for heart): ", [0, 1, 2])
        size = self.get_user_input(
            "Choose the size of the shape (0 to 5): ", range(6))
        x_coord = self.get_user_input(
            "Choose the X coordinate (0 to 31): ", range(32))
        y_coord = self.get_user_input(
            "Choose the Y coordinate (0 to 31): ", range(32))

        return [0, shape, size, 0, x_coord, y_coord]

    def plot_latent_space_helper(self):
        print("breakpoint 2")

        list_of_idx = []
        list_latent = []
        while True:
            latent_behavior = self.ask_user_input()
            list_latent.append(latent_behavior)
            list_of_idx.append(self.latent_to_index(latent_behavior))
            continue_choice = input(
                "Do you want to choose another shape? Enter 'N' to exit, any other key to continue: ").strip().upper()
            if continue_choice == 'N':
                break

        for i in range(len(list_of_idx)):
            print("for idx", list_of_idx[i], "for user input", list_latent[i])
            photo = self.dataloader.dataset[list_of_idx[i]]

            samples_zCx, params_zCx, decoder_output, mean, std = self._compute_q_zCx_single(
                self.dataloader, list_of_idx[i])
            print("mean", mean)
            print("std", std)
            print("*"*50)
            self.imshow(photo[0])
            self.imshow(decoder_output[0])
