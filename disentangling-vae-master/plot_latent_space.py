from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import itertools
import seaborn as sns
from scipy.stats import norm
import random
from matplotlib.colors import LinearSegmentedColormap

class LatentSpacePlotter:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        dataset_zip = np.load(
            'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
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
        with torch.no_grad():
            # Get the parameters of q(z|x) for the specified data point
            mean, log_var = self.model.encoder(x)
            params_zCx = (mean, log_var)

            # Sample z from q(z|x)
            # print("mean", mean.shape)
            # print("mean", mean)
            sample_zCx = self.model.reparameterize(mean, log_var)
            # print("sample_zCx", sample_zCx.shape)
            # print("sample_zCx", sample_zCx)

            output, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(
                sample_zCx)

        return sample_zCx, params_zCx, output, mean, log_var, lin_outputs, conv_outputs, conv_weights_tuple

    # def imshow(self, img):
    #     """Function to show an image."""
    #     # ... [rest of your existing method here]
    #     img = img / 2 + 0.5  # unnormalize if normalization was applied during pre-processing
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    def imshow(self, img_batch, captions=None, title=None):
        """Function to show images with captions.

        Args:
            img_batch (Tensor): A batch of images (tensor) of shape (N, C, H, W).
            captions (list of str): A list of captions (one for each image).

        """
        # Number of images
        num_images = len(img_batch)
        print("img_batch", img_batch)
        # Create a subplot
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

        for idx, img in enumerate(img_batch):
            # Unnormalize and convert to numpy
            img = img / 2 + 0.5

            if type(img) == torch.Tensor:
                npimg = img.numpy()
            npimg = np.transpose(npimg, (1, 2, 0))

            # Plotting
            if num_images == 1:
                axes.imshow(npimg)
                if captions:
                    axes.set_title(captions[idx])
                axes.axis('off')
            else:
                axes[idx].imshow(npimg)
                if captions:
                    axes[idx].set_title(captions[idx])
                axes[idx].axis('off')
        fig.suptitle(title)
        plt.tight_layout()
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
        rotation = self.get_user_input(
            "Choose the rotation of the shape (0 to 39): ", range(40))
        x_coord = self.get_user_input(
            "Choose the X coordinate (0 to 31): ", range(32))
        y_coord = self.get_user_input(
            "Choose the Y coordinate (0 to 31): ", range(32))

        return [0, shape, size, rotation, x_coord, y_coord]

    def experiment_one_vary_top_bottom_y_for_all_x(self):
        list_of_idx = []
        list_latent = []
        for x in range(32):
            for y in [0., 31.]:
                latent_behavior = [0., 1., 3., 0., x, y]
                list_of_idx.append(self.latent_to_index(latent_behavior))
                list_latent.append(latent_behavior)
        return list_of_idx, list_latent

    def experiment_two_hold_x_constant_incremently_increase_y(self, constant_x_value=0):
        list_of_idx = []
        list_latent = []
        for y in range(32):
            latent_behavior = [0., 1., 3., 0., constant_x_value, y]
            list_of_idx.append(self.latent_to_index(latent_behavior))
            list_latent.append(latent_behavior)
        return list_of_idx, list_latent

    def experiment_three_hold_x_constant_incremently_increase_y(self, constant_x_value=0, shape=0):
        list_of_idx = []
        list_latent = []
        for y in range(32):
            latent_behavior = [0., shape, 3., 0., constant_x_value, y]
            list_of_idx.append(self.latent_to_index(latent_behavior))
            list_latent.append(latent_behavior)
        return list_of_idx, list_latent

    def experiment_a_vary_x_y(self, shape=0):
        list_of_idx = []
        list_latent = []
        for x in range(32):
            for y in range(32):
                latent_behavior = [0., shape, 3., 0., x, y]
                list_of_idx.append(self.latent_to_index(latent_behavior))
                list_latent.append(latent_behavior)
        return list_of_idx, list_latent

    def experiment_five_generate_delta_vector(self, shape):
        # This function is going to hold shape constant, and generate all the other possible combinations of x and y, and size
        list_of_idx = []
        list_latent = []
        for x in range(32):
            for y in range(32):
                # for size in [3]:
                for size in range(6):
                    for rotation in range(40):
                        latent_behavior = [0., shape, size, rotation, x, y]
                        list_of_idx.append(
                            self.latent_to_index(latent_behavior))
                        list_latent.append(latent_behavior)
        return list_of_idx, list_latent

    def experiement_one_plot_helper(self, list_of_idx, list_latent, list_mean, list_std):
        class_info = list_latent[:, -1]
        print("class_info", class_info)
        print("list_mean", list_mean.shape)

        means_class_0 = list_mean[class_info == 0, :]
        means_class_31 = list_mean[class_info == 31, :]

        std_class_0 = list_std[class_info == 0, :]
        std_class_31 = list_std[class_info == 31, :]

        print("means_class_0", means_class_0.shape)
        print("means_class_31", means_class_31.shape)

        self.experiment_one_plot_all_data_latent(
            means_class_0, means_class_31, std_class_0, std_class_31)
        self.experiment_one_plot_subtraction_per_data_point(
            means_class_0, means_class_31, std_class_0, std_class_31)

    def experiment_two_plot_helper(self, list_of_idx, list_latent, list_mean, list_std):
        # list_mean, list_std, is going to contain all of the means and std
        # when we hold x to be constant, and then incremently increase y from 0 to 31, in that order

        class_info = list_latent[:, -1]
        print("class_info", class_info)
        assert all(class_info[i] <= class_info[i+1]
                   for i in range(len(class_info)-1)), "List is not in ascending order"

        print("list_mean", list_mean.shape)

        self.experiment_two_plot_all_data_latent(
            list_of_idx, list_latent, list_mean, list_std)
        self.experiment_two_plot_gaussian_latent(
            list_of_idx, list_latent, list_mean, list_std)

    def experiment_three_plot_helper(self, list_of_idx, list_latent, list_mean, list_std):
        self.experiment_three_plot_all_data_latent(
            list_of_idx, list_latent, list_mean, list_std
        )
        self.experiment_three_plot_gaussian_latent(
            list_of_idx, list_latent, list_mean, list_std
        )

    def experiment_four_plot_helper(self, list_of_idx, list_latent, list_mean, list_std, dim_manipulate, new_dim_value, decoder_output):

        print("ORIGINAL list_mean", list_mean)
        print("mean:, ", list_mean.shape)
        original_list_mean = list_mean[0].copy()
        for i in range(len(dim_manipulate)):
            list_mean[0][dim_manipulate[i]] = new_dim_value[i]

        print("NEW list_mean", list_mean)

        list_mean = torch.tensor(list_mean)
        manipulated_output = self.model.decoder(list_mean).detach()

        print("decoder_output", decoder_output.shape)
        print("manipulated_output", manipulated_output.shape)

        caption_original = f"{original_list_mean}(Original Image)"
        caption_manipulated = f"{list_mean[0].detach().numpy()}(Manipulated Image)"

        title = f"Original v.s. Manipulated {list_latent[0]}, for dimensions {dim_manipulate}, changed to {new_dim_value}"
        self.imshow([decoder_output[0], manipulated_output[0]], [
                    caption_original, caption_manipulated], title=title)

    def experiment_five_plot_helper(self, first_shape_list_of_idx, first_shape_list_latent, second_shape_list_of_idx, second_shape_list_latent):

        first_shape_mean = []
        first_shape_std = []
        second_shape_mean = []
        second_shape_std = []

        for i in range(len(first_shape_list_latent)):
            # print(
            #     "for idx", first_shape_list_of_idx[i], "for user input", first_shape_list_latent[i])
            # photo = self.dataloader.dataset[list_of_idx[i]]

            samples_zCx, params_zCx, decoder_output, mean, log_var, lin_out, conv_out, conv_weights_tuple = self._compute_q_zCx_single(
                self.dataloader, first_shape_list_of_idx[i])

            first_shape_std.append(np.array(np.sqrt(np.exp(log_var[0]))))
            first_shape_mean.append(np.array(mean[0]))

        for i in range(len(second_shape_list_latent)):
            # print("for idx", second_shape_list_of_idx[i],
            #       "for user input", second_shape_list_latent[i])
            # photo = self.dataloader.dataset[list_of_idx[i]]

            samples_zCx, params_zCx, decoder_output, mean, log_var, lin_out, conv_out, conv_weights_tuple = self._compute_q_zCx_single(
                self.dataloader, second_shape_list_of_idx[i])

            second_shape_std.append(np.array(np.sqrt(np.exp(log_var[0]))))
            second_shape_mean.append(np.array(mean[0]))

        first_shape_mean = np.array(first_shape_mean)
        second_shape_mean = np.array(second_shape_mean)

        first_shape_std = np.array(first_shape_std)
        second_shape_std = np.array(second_shape_std)

        print("first_shape_mean", first_shape_mean.shape)
        print("second_shape_mean", second_shape_mean.shape)
        first_shape_mean_avg = np.mean(first_shape_mean, axis=0)
        second_shape_mean_avg = np.mean(second_shape_mean, axis=0)

        first_shape_std_avg = np.mean(first_shape_std, axis=0)
        second_shape_std_avg = np.mean(second_shape_std, axis=0)

        delta_mean_vector, delta_std_vector = self.experiment_five_plot_average_latent_space(first_shape_list_latent, second_shape_list_latent,
                                                                                             first_shape_mean_avg, second_shape_mean_avg, first_shape_std_avg, second_shape_std_avg)

        self.experiment_five_plot_N_random_samples(first_shape_list_of_idx, first_shape_list_latent,
                                                   second_shape_list_of_idx, second_shape_list_latent, first_shape_mean, second_shape_mean, first_shape_std, second_shape_std, delta_mean_vector, delta_std_vector, N=10)

    def plot_neuron_heatmap(self, neuron_id, X, grid_size=(32, 32)):
        """
        Plots a heatmap for the specified neuron's activation values, 
        with the heat values normalized to be between -3 and +3 standard deviations.

        :param neuron_id: ID of the neuron to plot.
        :param X: Dictionary with neuron IDs as keys and lists of normalized outputs as values.
        :param grid_size: The dimensions of the grid (width, height).
        """
        # Extract the activation values for the given neuron
        activation_values = X.get(neuron_id, [])
        
        # Reshape the list into a 2D array
        activation_matrix = np.array(activation_values).reshape(grid_size)
        
        cmap = LinearSegmentedColormap.from_list(
        'custom_colormap', 
        [(0, 'darkblue'), (0.5, 'white'), (1, 'darkred')],
        N=256
        )

        # Plotting the heatmap
        plt.imshow(activation_matrix, cmap=cmap, interpolation='nearest', vmin=-3, vmax=3)
        # Plotting the heatmap
        plt.colorbar()
        plt.title(f"Heatmap for Neuron {neuron_id}, with shape 2, size 3, rotation 0")
        plt.savefig(f"../Experiment_results/experiment_a/shape_2/{neuron_id}_shape_2_size_3_rotation_0_heatmap.png")
        plt.clf()

    def experiment_a_plot_linear(self, map_lin1_out):
        lin1_means, lin1_stds = self.experiment_a_linear_layer_mean_std(map_lin1_out)
        A = map_lin1_out
        B = lin1_means
        C = lin1_stds
        X = {neuron_id: [] for neuron_id in range(256)}  # Initialize X with 256 neuron IDs

        for key in A:
            neuron_outputs = A[key][0]
            for neuron_id, output in enumerate(neuron_outputs):
                if C[0, neuron_id] != 0:  # To avoid division by zero
                    normalized_output = (output - B[0, neuron_id]) / C[0, neuron_id]
                else:
                    normalized_output = 0  # If standard deviation is 0, normalization is not meaningful
                X[neuron_id].append(normalized_output)

        # X is now the dictionary with neuron IDs as keys and lists of normalized outputs as values
        # Plotting the heatmap for neuron id
        for id in range(256):
            self.plot_neuron_heatmap(id, X)
        
    def experiment_a_plot_helper(self, map_lin1_out, map_lin2_out, map_lin3_out):
        # get the mean and std of the each neuron in the first linear layer
        lin1_means, lin1_stds = self.experiment_a_linear_layer_mean_std(map_lin1_out)
        # find the neurons that fire bigger than the mean + 2 std
        lin1_neurons_fired = self.experiment_a_find_neurons_fired(map_lin1_out, lin1_means, lin1_stds)
        # lin2_neurons_fired = self.experiment_a_find_neurons_fired(map_lin2_out, lin2_means, lin2_stds)
        # lin3_neurons_fired = self.experiment_a_find_neurons_fired(map_lin3_out, lin3_means, lin3_stds)

        self.experiment_a_plot_linear(map_lin1_out)

    def experiment_a_find_neurons_fired(self, map_lin_out, lin_means, lin_stds):
        # return a map of latent_behavior:[neurons_fired]
        # have neurons_fired in an array per experiment, (experiment_size, num_neurons_fired)
        map_neurons_fired = {key:[] for key in map_lin_out.keys()}

        assert len(map_neurons_fired.keys()) == 1024, "There should be 1024 experiments"
        # number of experiments
        for key, _ in map_lin_out.items():
            # number of neurons
            for j in range(lin_means.shape[1]):
                if map_lin_out[key][:, j] > lin_means[0, j] + 2 * lin_stds[0, j]:
                    # append the index of the neuron 
                    map_neurons_fired[key].append(j)
        return map_neurons_fired

    def experiment_a_linear_layer_mean_std(self, map_lin_out):
        # returns mean and std for all neurons across the experiments 
        # turn map_lin_out items into an array
        list_lin_out = np.stack(list(map_lin_out.values()))
        lin_means = np.mean(list_lin_out, axis=0)
        lin_stds = np.std(list_lin_out, axis=0)
        return lin_means, lin_stds
    
    def experiment_one_plot_subtraction_per_data_point(self, means_class_0, means_class_31, std_class_0, std_class_31):
        # This function is going to plot the subtracted difference, per each specific x, the difference in means and standard deviations for y = 0 and y = 31
        assert means_class_0.shape == means_class_31.shape
        assert std_class_0.shape == std_class_31.shape

        for i in range(10):
            x_values = np.full(means_class_0.shape[0], i)

            plt.scatter(
                x_values, means_class_31[:, i] - means_class_0[:, i], color='blue', label='Difference in Means' if i == 0 else "")

        plt.title(
            f'Difference in Means for y=0 and y=31 across all x simultaneously for all 10 latent dimensions, N = {means_class_0.shape[0]}')
        plt.xlabel('Dimension')
        plt.ylabel('Difference in Mean Value')
        # Set x-axis labels for dimensions
        plt.xticks(range(10), [f'Dim {i+1}' for i in range(10)])
        plt.legend()
        plt.show()

        for i in range(10):
            x_values = np.full(std_class_0.shape[0], i)
            plt.scatter(
                x_values, std_class_31[:, i] - std_class_0[:, i], color='blue', label='Difference in Standard Deviations' if i == 0 else "")
        plt.title(
            f'Difference in STD for y=0 and y=31 across all x simultaneously for all 10 latent dimensions, N = {means_class_0.shape[0]}')

        plt.xlabel('Dimension')
        plt.ylabel('Difference in Standard Deviation Value')
        # Set x-axis labels for dimensions
        plt.xticks(range(10), [f'Dim {i+1}' for i in range(10)])
        plt.legend()
        plt.show()

    def experiment_one_plot_all_data_latent(self, means_class_0, means_class_31, std_class_0, std_class_31):
        # This function simply plots the means and standard deviations for all 10 latent dimensions for all 32 x values
        for i in range(10):
            x_values_class_0 = np.full(means_class_0.shape[0], i) - 0.1
            x_values_class_31 = np.full(means_class_31.shape[0], i) + 0.1

            plt.scatter(
                x_values_class_0, means_class_0[:, i], color='red', label='Y=0' if i == 0 else "")
            plt.scatter(
                x_values_class_31, means_class_31[:, i], color='blue', label='Y=31' if i == 0 else "")

        plt.title(
            f'Comparison of for y=0 and y=31 across all x simultaneously for all 10 latent dimensions, N = {x_values_class_0.shape[0] + x_values_class_31.shape[0]}')
        plt.xlabel('Dimension')
        plt.ylabel('Mean Value')
        # Set x-axis labels for dimensions
        plt.xticks(range(10), [f'Dim {i+1}' for i in range(10)])
        plt.legend()
        plt.show()

        for i in range(10):
            x_values_class_0 = np.full(std_class_0.shape[0], i) - 0.1
            x_values_class_31 = np.full(std_class_31.shape[0], i) + 0.1

            plt.scatter(
                x_values_class_0, std_class_0[:, i], color='red', label='Y=0' if i == 0 else "")
            plt.scatter(
                x_values_class_31, std_class_31[:, i], color='blue', label='Y=31' if i == 0 else "")

        plt.title(
            f'Comparison of STD for y=0 and y=31 across all x simultaneously for all 10 latent dimensions, N = {x_values_class_0.shape[0] + x_values_class_31.shape[0]}')
        plt.xlabel('Dimension')
        plt.ylabel('Standard Deviation Value')
        # Set x-axis labels for dimensions
        plt.xticks(range(10), [f'Dim {i+1}' for i in range(10)])
        plt.legend()
        plt.show()

    def experiment_two_plot_all_data_latent(self, list_of_idx, list_latent, list_mean, list_std):
        num_dimensions = list_mean.shape[1]
        y_values = list_mean.shape[0]  # Assuming y values range from 0 to 31

        assert y_values == 32, "Y values should be 32"
        assert num_dimensions == 10, "Number of dimensions should be 10"

        # Light to dark blue colors
        colors = plt.cm.Blues(np.linspace(0.2, 1, 32))

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot for Mean
        ax1.set_title(
            f'For X = {int(list_latent[0,-2])}, Y varies from Y=0 to Y=31 for Mean')
        ax1.set_xlabel('Dimension Index')
        ax1.set_ylabel('Value (Mean)')
        for dim in range(num_dimensions):
            for y in range(32):
                ax1.plot(dim, list_mean[y, dim], 'o', color=colors[y], label=f'y={y}' if y in [
                         0, 31] and dim == 0 else "")

        # Plot for Standard Deviation
        ax2.set_title(
            f'For X = {int(list_latent[0,-2])}, Y varies from Y=0 to Y=31 for STD')
        ax2.set_xlabel('Dimension Index')
        ax2.set_ylabel('Value (STD)')
        for dim in range(num_dimensions):
            for y in range(32):
                ax2.plot(dim, list_std[y, dim], 'o', color=colors[y], label=f'y={y}' if y in [
                         0, 31] and dim == 0 else "")

        # Adjust the layout and show the plot
        fig.suptitle('Ellipse, size 3, rotation 0')
        plt.tight_layout()
        plt.show()

    def experiment_two_plot_gaussian_latent(self, list_of_idx, list_latent, list_mean, list_std):
        num_dimensions = list_mean.shape[1]
        y_values = list_mean.shape[0]  # Assuming y values range from 0 to 31

        assert y_values == 32, "Y values should be 32"
        assert num_dimensions == 10, "Number of dimensions should be 10"

        # Light to dark blue colors
        colors = plt.cm.Blues(np.linspace(0.2, 1, 32))

        # X range for plotting Gaussian curves
        x_range = np.linspace(-3, 3, 1000)

        # Create a figure with one subplot per dimension
        fig, axes = plt.subplots(
            num_dimensions, 1, figsize=(10, 1.15 * num_dimensions))

        # Check if there is only one dimension (axes won't be an array in this case)
        if num_dimensions == 1:
            axes = [axes]

        for dim in range(num_dimensions):
            ax = axes[dim]
            ax.set_title(f'Gaussian Curves for Dimension {dim}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')

            for y in range(32):

                mu = list_mean[y, dim]
                sigma = list_std[y, dim]
                ax.plot(x_range, norm.pdf(x_range, mu, sigma),
                        color=colors[y], label=f'y={y}' if y in [0, 31] else "")

            if dim == 0:
                ax.legend()
        fig.suptitle(f'Ellipse, size 3, rotation 0, x = {list_latent[0,-2]}')
        plt.tight_layout()
        plt.show()

    def experiment_three_plot_all_data_latent(self, list_of_idx, list_latent, list_mean, list_std):
        num_dimensions = list_mean.shape[1]
        y_values = 32

        # grab total different x_Values entered
        num_x_values_entered = len(list_latent) // y_values

        for x_value in range(num_x_values_entered):
            cur_x_index = 32*x_value
            # Light to dark blue colors
            colors = plt.cm.Blues(np.linspace(0.2, 1, 32))

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # Plot for Mean
            ax1.set_title(
                f'For X = {int(list_latent[cur_x_index,-2])}, Y varies from Y=0 to Y=31 for Mean')
            ax1.set_xlabel('Dimension Index')
            ax1.set_ylabel('Value (Mean)')
            for dim in range(num_dimensions):
                for y in range(32):
                    ax1.plot(dim, list_mean[cur_x_index + y, dim], 'o', color=colors[y], label=f'y={y}' if y in [
                        0, 31] and dim == 0 else "")

            # Plot for Standard Deviation
            ax2.set_title(
                f'For X = {int(list_latent[cur_x_index,-2])}, Y varies from Y=0 to Y=31 for STD')
            ax2.set_xlabel('Dimension Index')
            ax2.set_ylabel('Value (STD)')

            for dim in range(num_dimensions):
                for y in range(32):
                    ax2.plot(dim, list_std[cur_x_index + y, dim], 'o', color=colors[y], label=f'y={y}' if y in [
                        0, 31] and dim == 0 else "")

            # Adjust the layout and show the plot
            fig.suptitle(f'Shape={int(list_latent[0][1])}, size 3, rotation 0')

            plt.tight_layout()
            file_path = f"../Experimental_Results/experiment_three/shape_{int(list_latent[0][1])}/x_{int(list_latent[cur_x_index,-2])}.png"
            plt.savefig(file_path)

    def experiment_three_plot_gaussian_latent(self, list_of_idx, list_latent, list_mean, list_std):
        num_dimensions = list_mean.shape[1]
        y_values = 32

        # grab total different x_Values entered
        num_x_values_entered = len(list_latent) // y_values

        for x_value in range(num_x_values_entered):

            cur_x_index = 32*x_value
            colors = plt.cm.Blues(np.linspace(0.2, 1, 32))
            # X range for plotting Gaussian curves
            x_range = np.linspace(-3, 3, 1000)

            # Create a figure with one subplot per dimension
            fig, axes = plt.subplots(
                num_dimensions, 1, figsize=(10, 1.15 * num_dimensions))

            # Check if there is only one dimension (axes won't be an array in this case)
            if num_dimensions == 1:
                axes = [axes]

            for dim in range(num_dimensions):
                ax = axes[dim]
                ax.set_title(f'Gaussian Curves for Dimension {dim}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Probability Density')

                for y in range(32):

                    mu = list_mean[cur_x_index + y, dim]
                    sigma = list_std[cur_x_index + y, dim]
                    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
                            color=colors[y], label=f'y={y}' if y in [0, 31] else "")

                if dim == 0:
                    ax.legend()
            fig.suptitle(
                f'Shape = {int(list_latent[0][1])}, size 3, rotation 0, x = {int(list_latent[cur_x_index,-2])}')
            plt.tight_layout()
            file_path = f"../Experimental_Results/experiment_three/shape_{int(list_latent[0][1])}/x_{int(list_latent[cur_x_index,-2])}_g.png"
            plt.savefig(file_path)

    def experiment_five_plot_average_latent_space(self, first_shape_list_latent, second_shape_list_latent, first_shape_mean_avg, second_shape_mean_avg, first_shape_std_avg, second_shape_std_avg):
        # This function is going to take the mean_avg & std_avg of the two shapes, and plot them as gaussian curves

        # print("first_shape_mean_avg", first_shape_mean_avg)
        # print("first_shape_list_latent", first_shape_list_latent)
        num_dimensions = len(first_shape_mean_avg)
        y_values = 2

        # X range for plotting Gaussian curves
        x_range = np.linspace(-3, 3, 1000)

        # Create a figure with one subplot per dimension
        fig, axes = plt.subplots(
            num_dimensions, 1, figsize=(10, 1.15 * num_dimensions))

        # Check if there is only one dimension (axes won't be an array in this case)
        if num_dimensions == 1:
            axes = [axes]

        for dim in range(num_dimensions):
            ax = axes[dim]
            ax.set_title(f'Gaussian Curves for Dimension {dim}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability Density')

            first_mu = first_shape_mean_avg[dim]
            first_sigma = first_shape_std_avg[dim]
            ax.plot(x_range, norm.pdf(x_range, first_mu, first_sigma),
                    color='red', label=f'First Shape = {first_shape_list_latent[0][1]}')

            second_mu = second_shape_mean_avg[dim]
            second_sigma = second_shape_std_avg[dim]
            ax.plot(x_range, norm.pdf(x_range, second_mu, second_sigma),
                    color='blue', label=f'Second Shape = {second_shape_list_latent[0][1]}')

            if dim == 0:
                ax.legend()
        fig.suptitle(
            f'Comparsion of the average of all the possible combinations for these two different shapes')
        plt.tight_layout()
        plt.show()

        delta_vector_mean = first_shape_mean_avg - second_shape_mean_avg
        delta_vector_std = first_shape_std_avg - second_shape_std_avg

        return delta_vector_mean, delta_vector_std

    def experiment_five_plot_N_random_samples(self, first_shape_list_of_idx, first_shape_list_latent,
                                              second_shape_list_of_idx, second_shape_list_latent, first_shape_mean, second_shape_mean, first_shape_std, second_shape_std, delta_mean_vector, delta_std_vector, N=10):
        # This function is going to sample 5 random samples from the first_shape, subtract the delta vector for the latent space,
        # sample from the updated latent space, and then plot original image, the original reconstructed image,
        # and the reconstructed image after applying the delta vector.
        # repeat for the second shape.

        # generate N//2 random numbers between the 0-index of the first shape list of idx and the last index of the first shape list of idx
        random_first_shape_idx = random.sample(
            range(0, len(first_shape_list_of_idx)), N//2)

        for idx in random_first_shape_idx:

            original_input_photo = self.dataloader.dataset[first_shape_list_of_idx[idx]]
            print("original_input_photo", original_input_photo)

            mean = first_shape_mean[idx]
            log_var = second_shape_std[idx]
            sample_zCx = self.model.reparameterize(mean, log_var)

            # convert sample_zCx from a numpy N vector to a 1XN tensor array
            sample_zCx = torch.tensor(sample_zCx).unsqueeze(0)
            print("original sample_zCx", sample_zCx)
            original_output_image, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(
                sample_zCx)

            original_output_image = original_output_image.detach()[0]

            # apply the delta vector to the latent space
            mean = mean - delta_mean_vector
            # log_var = log_var - delta_std_vector
            # sample_zCx = self.model.reparameterize(mean, log_var)
            # sample_zCx = torch.tensor(sample_zCx).unsqueeze(0)
            sample_zCx = torch.tensor(mean).unsqueeze(0)
            print("modified sample_zCx", sample_zCx)
            manipulated_image, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(
                sample_zCx)

            manipulated_image = manipulated_image.detach()[0]
            print("manipulated_image", manipulated_image)

            self.imshow([original_output_image, manipulated_image], [
                        "Original Reconstructed Image", "Manipulated Image"], title=f"Original Shape = {first_shape_list_latent[idx][1]}, Intended Manipulated Shape = {second_shape_list_latent[idx][1]}")

        random_second_shape_idx = random.sample(
            range(0, len(second_shape_list_of_idx)), N//2)

        for idx in random_second_shape_idx:

            original_input_photo = self.dataloader.dataset[second_shape_list_of_idx[idx]]
            print("original_input_photo", original_input_photo)

            mean = second_shape_mean[idx]
            log_var = second_shape_std[idx]
            sample_zCx = self.model.reparameterize(mean, log_var)

            # convert sample_zCx from a numpy N vector to a 1XN tensor array
            sample_zCx = torch.tensor(sample_zCx).unsqueeze(0)
            print("unmodified sample_zCx", sample_zCx)
            original_output_image, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(
                sample_zCx)

            original_output_image = original_output_image.detach()[0]

            # apply the delta vector to the latent space
            mean = mean + delta_mean_vector
            # log_var = log_var + delta_std_vector
            # sample_zCx = self.model.reparameterize(mean, log_var)
            # sample_zCx = torch.tensor(sample_zCx).unsqueeze(0)
            sample_zCx = torch.tensor(mean).unsqueeze(0)
            print("modified sample_zCx", sample_zCx)
            manipulated_image, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(
                sample_zCx)

            manipulated_image = manipulated_image.detach()[0]

            print("manipulated_image", manipulated_image)

            self.imshow([original_output_image, manipulated_image], [
                        "Original Reconstructed Image", "Manipulated Image"], title=f"Original Shape = {second_shape_list_latent[idx][1]}, Intended Manipulated Shape = {first_shape_list_latent[idx][1]}")

    def experiment_b_all_features(self):
        # This function is going to generate all possible combinations of features 
        # it outputs list_of_idx, list_latent
        list_of_idx = []
        list_latent = []
        for shape in range(3):
            for size in range(6):
                for rotation in range(40):
                    for x in range(32):
                        for y in range(32):
                            latent_behavior = [0., shape, size, rotation, x, y]
                            list_of_idx.append(
                                self.latent_to_index(latent_behavior))
                            list_latent.append(latent_behavior)
        assert(len(list_of_idx) == 3 * 6 * 40 * 32 * 32)
        return list_of_idx, list_latent

    def experiment_b_save_linear_layer_outputs(self, map_lin1_out):
        # this function is going to save the linear layer outputs for each latent_behavior 
        # it also generates the mean and std for each neuron and saves it separately
        # map_lin1_out is a dictionary with key:latent_behavior, value:lin1_out (1, 256)
        print("saving mean, std and lin1_out")
        mean, std = self.experiment_a_linear_layer_mean_std(map_lin1_out)
        assert(mean.shape == (1, 256))
        assert(std.shape == (1, 256))
        np.save("Experiment_results/experiment_b/lin1_out.npy", map_lin1_out)
        np.save("Experiment_results/experiment_b/lin1_mean.npy", mean)
        np.save("Experiment_results/experiment_b/lin1_std.npy", std)

    def experiment_c_neuron_firing_map_helper(self, map_lin_out, mean, std, x_val):
        total_count =0
        map_fired_count = { key : 0 for key in range(256) } # key: neuron id, value: number of times it has fired
        for feat in map_lin_out.keys():
            # key is a string of the latent behavior
            # convert to a list of floats
            list_feat = list(map(float, feat[1:-1].split(',')))
            if list_feat[-2] == x_val:
                total_count += 1
                for neuron_id, output in enumerate(map_lin_out[feat][0]):
                    if output > mean[0, neuron_id] + 1 * std[0, neuron_id]:
                        map_fired_count[neuron_id] += 1
        
        # sort the map by the number of neurons fired
        sorted_map_fired_count = sorted(map_fired_count.items(), key=lambda x: x[1], reverse=True)
        print("sorted_map_fired_count", sorted_map_fired_count)
        # print the top 10 neurons that have fired and the percentage they have fired
        for i in range(10):
            print(f"Neuron {sorted_map_fired_count[i][0]} has fired {sorted_map_fired_count[i][1]} times, which is {sorted_map_fired_count[i][1] / total_count * 100}% of the time")
        return sorted_map_fired_count
    
    def experiment_c_x_val_neuron_fired(self, x_val):
        """
        This function loads the presaved linear layer outputs, mean, and std,
        find the top 10 neurons that have fired and the percentage they have fired
        across all possible combinations of features
        :param x_val : a value in between 0 and 31 that specifies the x position
        """
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)
        return self.experiment_c_neuron_firing_map_helper(map_lin_out, mean, std, x_val)
    
    def experiment_d_neuron_idx_helper(self, map_lin_out, mean, std, neuron_index):
        """
        Works the same as experiment_c_helper but for a specific neuron index and more efficiently 
        by storing the specific neuron's firing pattern in a dictionary
        """
        total_count = 1 * 32 * 40 * 6 * 3
        map_fired_count_ind = { key : 0 for key in range(32) } # key: x_val, value: number of times the neuron has fired
        for x in range(32):
            print("x", x)
            for feat in map_lin_out.keys():
                # key is a string of the latent behavior
                # convert to a list of floats
                list_feat = list(map(float, feat[1:-1].split(',')))
                if list_feat[-2] == x:
                    if map_lin_out[feat][0][neuron_index] > mean[0, neuron_index] + 1 * std[0, neuron_index]:
                        map_fired_count_ind[x] += 1
        map_fired_percentage_ind = { key : value / total_count * 100 for key, value in map_fired_count_ind.items() }
        return map_fired_percentage_ind

    def experiment_d_neuron_idx_fired_for_all_x(self, neuron_index):
        """
        This function runs experiment_c_x_val_neuron_fired for all x values and graphs the indexed neuron's
        firing pattern across all x values
        :param neuron_index : the index of the neuron to graph
        """
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)

        map_fired_percentage = self.experiment_d_neuron_idx_helper(map_lin_out, mean, std, neuron_index)
        print("map_fired_percentage_ind", map_fired_percentage)
        # put the map_fired_percentage_ind.values() in order of x_val
        map_fired_percentage = {key: map_fired_percentage[key] for key in sorted(map_fired_percentage.keys())}
        plt.plot(range(32), map_fired_percentage.values())
        plt.title(f"Neuron {neuron_index} Firing Pattern Across All X Values")
        plt.xlabel("X Value")
        plt.ylabel("Firing Percentage")
        plt.show()

    def experiment_e_neuron_fired_z_score(self, x_val):
        """
        This function repeats the purpose of experiment c, where for a given x_val, it finds the neurons that fired
        and calculates the z-score of its output value 
        """
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)
        map_fired_z_score = { key : 0 for key in range(256) } # key: neuron id, value: number of times it has fired
        for feat in map_lin_out.keys():
            # key is a string of the latent behavior
            # convert to a list of floats
            list_feat = list(map(float, feat[1:-1].split(',')))
            if list_feat[-2] == x_val:
                for neuron_id, output in enumerate(map_lin_out[feat][0]):
                    map_fired_z_score[neuron_id] += (output - mean[0, neuron_id]) / std[0, neuron_id]
        # calculate mean z_score
        map_fired_z_score = {key: value / (32 * 40 * 6 * 3) for key, value in map_fired_z_score.items()}
        # sort the map by z_score
        sorted_map_fired_z_score = sorted(map_fired_z_score.items(), key=lambda x: x[1], reverse=True)
        print("sorted_map_fired_z_score", sorted_map_fired_z_score)
        # print the top 10 neurons with the highest z scores
        for i in range(10):
            print(f"Neuron {sorted_map_fired_z_score[i][0]} has an average z-score of {sorted_map_fired_z_score[i][1]}")
        return sorted_map_fired_z_score 
    
    def experiment_f_neuron_idx_helper(self, map_lin_out, mean, std, neuron_index):
        """
        Works the same as experiment_e_helper but for a specific neuron index and more efficiently 
        by storing the specific neuron's firing pattern in a dictionary
        """
        map_fired_z_score_ind = { key : 0 for key in range(32) } # key: x_val, value: number of times the neuron has fired
        for x in range(32):
            print("x", x)
            for feat in map_lin_out.keys():
                # key is a string of the latent behavior
                # convert to a list of floats
                list_feat = list(map(float, feat[1:-1].split(',')))
                if list_feat[-2] == x:
                    map_fired_z_score_ind[x] += (map_lin_out[feat][0][neuron_index] - mean[0, neuron_index]) / std[0, neuron_index]
        map_fired_z_score_ind = { key : value / (32 * 40 * 6 * 3) for key, value in map_fired_z_score_ind.items() }
        return map_fired_z_score_ind

    def experiment_f_neuron_idx_fired_for_all_x_z_score(self, neuron_index):
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)
        map_neuron_z_score = self.experiment_f_neuron_idx_helper(map_lin_out, mean, std, neuron_index)
        # plot the z_score for each x_val
        map_neuron_z_score = {key: map_neuron_z_score[key] for key in sorted(map_neuron_z_score.keys())}
        plt.plot(range(32), map_neuron_z_score.values())
        plt.title(f"Neuron {neuron_index} Z-Score Across All X Values")
        plt.xlabel("X Value")
        plt.ylabel("Z-Score")
        plt.show()

    def experiment_g_prob_helper(self, map_lin_out, mean, std, neuron_ind, x_val):
        total_count = 32 * 32 * 40 * 6 * 3
        total_count_given_x = 32 * 40 * 6 * 3
        neuron_fires_count = 0
        neuron_fires_count_given_x = 0
        x_val_given_neuron_fires = 0

        for feat in map_lin_out.keys():
            # key is a string of the latent behavior
            # convert to a list of floats
            list_feat = list(map(float, feat[1:-1].split(',')))
            if list_feat[-2] == x_val:
                if map_lin_out[feat][0][neuron_ind] > mean[0, neuron_ind] + 1 * std[0, neuron_ind]:
                    neuron_fires_count_given_x += 1
            if map_lin_out[feat][0][neuron_ind] > mean[0, neuron_ind] + 1 * std[0, neuron_ind]:
                neuron_fires_count += 1
                if list_feat[-2] == x_val:
                    x_val_given_neuron_fires += 1
        prob_neuron_fires = neuron_fires_count / total_count
        prob_neuron_fires_given_x_val = neuron_fires_count_given_x / total_count_given_x
        prob_x_val_given_neuron_fires = x_val_given_neuron_fires / neuron_fires_count
        return prob_neuron_fires, prob_neuron_fires_given_x_val, prob_x_val_given_neuron_fires

    def experiment_g_neuron_idx_x_val_bayes(self, neuron_index, x_val):
        """
        This function calculates the probability that the neuron fires on average, the probability that the neuron fires given that X value,
        and the probability that x is that value given that the neuron fires
        """
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)
        prob_x_val = 1 / 32
        prob_neuron_fires, prob_neuron_fires_given_x_val, prob_x_val_given_neuron_fires = self.experiment_g_prob_helper(map_lin_out, mean, std, neuron_index, x_val)

        print(f"Probability that neuron {neuron_index} fires on average: {prob_neuron_fires}")
        print(f"Probability that neuron {neuron_index} fires given that x is {x_val}: {prob_neuron_fires_given_x_val}")
        print(f"Probability that x is {x_val} given that neuron {neuron_index} fires: {prob_x_val_given_neuron_fires}")
        print(f"P(x={x_val} | neuron_fires) = P(neuron_fires | x={x_val}) * P(x={x_val}) / P(neuron_fires given guassian curve) = {prob_neuron_fires_given_x_val} * {prob_x_val} / {0.158} = {prob_neuron_fires_given_x_val * prob_x_val / 0.158}")
    
    def experiment_h_helper(self, map_lin_out, mean, std, neuron_ind):
        total_count = 32 * 32 * 40 * 6 * 3
        total_count_given_x = 32 * 40 * 6 * 3
        map_neuron_fires_count = { key : 0 for key in range(32) } # key: x_val, value: number of times the neuron has fired
        map_neuron_fires_count_given_x = { key : 0 for key in range(32) } # key: x_val, value: number of times the neuron has fired
        map_x_val_given_neuron_fires = { key : 0 for key in range(32) } # key: x_val, value: number of times the neuron has fired
        for feat in map_lin_out.keys():
            # key is a string of the latent behavior
            # convert to a list of floats
            list_feat = list(map(float, feat[1:-1].split(',')))
            x_val  = list_feat[-2]

            if map_lin_out[feat][0][neuron_ind] > mean[0, neuron_ind] + 1 * std[0, neuron_ind]:
                map_neuron_fires_count[x_val] += 1
                map_neuron_fires_count_given_x[x_val] += 1
                map_x_val_given_neuron_fires[x_val] += 1
        return

    def experiment_h(self, neuron_ind):
        """
        obtain the groundtruth probability of (X = val | neuron fires)
        as well as the calculated probability assuming that the neuron fires like a gaussain curve
        plot the differences in the probabilities for each X value
        """
        map_lin_out_path = "Experiment_results/experiment_b/lin1_out.npy"
        map_lin_out = np.load(map_lin_out_path, allow_pickle=True).item()
        mean_path = "Experiment_results/experiment_b/lin1_mean.npy"
        mean = np.load(mean_path)
        std_path = "Experiment_results/experiment_b/lin1_std.npy"
        std = np.load(std_path)
        prob_x_val = 1 / 32
        # same logic as experiment g helper but use a map for all x values
        map_gc_probabilities, map_calculated_probabilities = self.experiment_h_helper(map_lin_out, mean, std, neuron_ind)
        # plot the differences in the probabilities for each X value


        return 
    def main_experiment(self):

        list_of_idx = []
        list_latent = []
        list_mean = []
        list_std = []
        map_lin1_out = {}
        map_lin2_out = {}
        map_lin3_out = {}

        running_experiment = False
        pass_decoder = False
        while True:
            prompt = " Enter the experiment number, else enter N \n" + \
                "(Experiment 1): Incremently increase the x from 0 to 31, while measuring the difference between the \n" +  \
                "latent dimensions for each x when y = 0 v.s. y = 31 for the shape ellipse, rotation 0, and size 3?\n" + \
                "(Experiment 2): Hold X to be constant at 0, and incremently increase the y from 0 to 31, while visualizing the difference in latent space, for the shape ellipse, rotation 0, and size 3 \n" + \
                "(Experiment 3): Repetition of Experiment 2, except you can control shape, and input multiple possible X \n" +\
                "(Experiment 4): Manipulation of multiple latent dimensions \n\n" +\
                "(Experiment 5): Generating the delta vector between two shapes (hold rotation constant)" +\
                "(Experiment a): Hold shape, scale, and rotation constant, and vary (x, y) position each from 0 to 31, while measuring the mean and std of neurons in the three linear layers. \n" + \
                "Generate a heatmap for each neuron in each linear layer, with the heat values normalized to be between -3 and +3 standard deviations. \n" + \
                "NEED TO MANUALLY CHANGE SHAPE FOLDER IN EXPERIMENT A PLOT HELPER \n" + \
                "(Experiment b): Generate the first linear layer output for all combinations of features\n" + \
                "(Experiment c): Given all possible latent behaviors, input an X value, and find the neurons that have fired for that X generalized to all combinations of features \n" + \
                "(Experiment d): Given a neuron number, repeat experiment c for all X values and plot firing percentage\n" + \
                "(Experiment e): Similar to experiment c, input an X value, and find the neurons that fired for that X, but instead of counting the number of times it fired, \n" + \
                "calculate the z-score of the number of times it fired, and plot the z-score for all neurons \n" + \
                "(Experiment f): Given a neuron number, repeat experiment e for all X values and plot average z_score\n" + \
                "(Experiment g): Given a neuron number, and an X value, calculate the probability that the neuron fires on average, the probability that the neuron fires given that X value, \n" + \
                "and the probability that x is that value given that the neuron fires \n" + \
                "(Experiment h): Repeat experiment g for all X values of a given neuron and plot the probability that x is that value given that the neuron fires \n" + \
                "Experiment Number (ENTER): "

            automatic_choice = input(
                prompt).strip()
            if automatic_choice == '1':
                list_of_idx, list_latent = self.experiment_one_vary_top_bottom_y_for_all_x()
                running_experiment = '1'
                pass_decoder = True
                break
            elif automatic_choice == '2':
                prompt = "What value do you want to hold X to be constant at? \n (ENTER):"
                try:
                    x_value = int(input(
                        prompt).strip().upper())
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue

                list_of_idx, list_latent = self.experiment_two_hold_x_constant_incremently_increase_y(
                    x_value)
                running_experiment = '2'
                pass_decoder = True
                break
            elif automatic_choice == '3':
                prompt = "What values of X do you want to hold it to be constant at? Please enter them separated by a comma \n (ENTER): "
                try:
                    x_values = input(prompt).strip()
                    x_values = list(map(int, x_values.split(',')))
                except ValueError:
                    print("Invalid input. Please enter numbers separated by a comma.")

                prompt = "What shape do you want to choose? Enter 0 for square, 1 for ellipse, 2 for heart: \n (ENTER): "
                try:
                    shape = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")

                list_of_idx = []
                list_latent = []
                for x_value in x_values:
                    cur_list_of_idx, cur_list_latent = self.experiment_three_hold_x_constant_incremently_increase_y(
                        x_value, shape)
                    list_of_idx.extend(cur_list_of_idx)
                    list_latent.extend(cur_list_latent)
                running_experiment = '3'
                pass_decoder = True
                break
            elif automatic_choice == 'a':
                prompt = "What shape do you want to choose? Enter 0 for square, 1 for ellipse, 2 for heart: \n (ENTER): "
                try:
                    shape = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
                list_of_idx, list_latent = self.experiment_a_vary_x_y(shape)
                running_experiment = 'a'
                pass_decoder = True
                break
            elif automatic_choice == '4':
                running_experiment = '4'
                print("Please Enter the Original Desired Image Information as below \n")
                latent_behavior = self.ask_user_input()
                list_of_idx.append(self.latent_to_index(latent_behavior))
                list_latent.append(latent_behavior)

                dim_manipulate = []
                new_dim_value = []
                while True:
                    prompt = "Out of the 10 latent dimensions, which one do you want to manipulate? Enter a number from 0 to 9: \n (ENTER): "
                    try:
                        dim_manipulate.append(int(input(prompt).strip()))
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    prompt = "What would you want the new value of that latent dimension be? Enter a number between -3 to 3 \n (ENTER): "
                    try:
                        new_dim_value.append(float(input(prompt).strip()))
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                    # Ask if the user wants to continue
                    continue_choice = input(
                        "Do you want to manipulate another latent dimension? Enter 'N' to exit, any other key to continue:\n ").strip().upper()
                    if continue_choice == 'N':
                        break
                pass_decoder = True
                break
            elif automatic_choice == '5':
                running_experiment = '5'

                prompt = "Enter the two shapes you want to generate the delta vector for \n" + \
                    "Enter at least two, and separate them by a comma \n" + \
                    "Enter 0 for square, 1 for ellipse, 2 for heart \n" +\
                    "(ENTER): "

                try:
                    x_values = input(prompt).strip()
                    x_values = list(map(int, x_values.split(',')))
                except ValueError:
                    print("Invalid input. Please enter numbers separated by a comma.")

                first_shape = x_values[0]
                second_shape = x_values[1]

                first_shape_list_of_idx, first_shape_list_latent = self.experiment_five_generate_delta_vector(
                    first_shape)
                second_shape_list_of_idx, second_shape_list_latent = self.experiment_five_generate_delta_vector(
                    second_shape)
                break
            elif automatic_choice == 'b':
                running_experiment = 'b'
                list_of_idx, list_latent = self.experiment_b_all_features()
                pass_decoder = True
                break
            elif automatic_choice == 'c':
                running_experiment = 'c'
                prompt = "Enter the X value you want to hold constant \n" + \
                    "(ENTER): "
                try:
                    x_value = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
                break 
            elif automatic_choice == 'd':
                running_experiment = 'd'
                prompt = "Enter the neuron index you want to investigate \n" + \
                    "(ENTER): "
                try:
                    neuron_idx = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number in between 0 and 256.")
                break
            elif automatic_choice == 'e':
                running_experiment = 'e'
                prompt = "Enter the X value you want to hold constant \n" + \
                    "(ENTER): "
                try:
                    x_value = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
                break
            elif automatic_choice == 'f':
                running_experiment = 'f'
                prompt = "Enter the neuron index you want to investigate \n" + \
                    "(ENTER): "
                try:
                    neuron_idx = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number in between 0 and 256.")
                break
            elif automatic_choice == 'g':
                running_experiment = 'g'
                prompt = "Enter the X value you want to hold constant \n" + \
                    "(ENTER): "
                try:
                    x_value = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
                prompt = "Enter the neuron index you want to investigate \n" + \
                    "(ENTER): "
                try:
                    neuron_idx = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number in between 0 and 256.")
                break
            elif automatic_choice == 'h':
                running_experiment = 'h'
                prompt = "Enter the neuron index you want to investigate \n" + \
                    "(ENTER): "
                try:
                    neuron_idx = int(input(prompt).strip())
                except ValueError:
                    print("Invalid input. Please enter a number in between 0 and 256.")
                break

            latent_behavior = self.ask_user_input()
            list_latent.append(latent_behavior)
            list_of_idx.append(self.latent_to_index(latent_behavior))
            continue_choice = input(
                "Do you want to choose another shape? Enter 'N' to exit, any other key to continue: ").strip().upper()
            if continue_choice == 'N':
                break

        if pass_decoder:
            print("running_experimet", running_experiment)
            for i in range(len(list_of_idx)):
                # print("for idx", list_of_idx[i],
                #       "for user input", list_latent[i])
                photo = self.dataloader.dataset[list_of_idx[i]]
                samples_zCx, params_zCx, decoder_output, mean, log_var, lin_out, conv_out, conv_weights_tuple = self._compute_q_zCx_single(
                    self.dataloader, list_of_idx[i])

                # self.imshow(decoder_output)

                std = np.array(np.sqrt(np.exp(log_var[0])))
                mean = np.array(mean[0])

                list_mean.append(mean)
                list_std.append(std)
                if running_experiment == 'a':
                    key = str(list_latent[i][-2:])
                else:
                    key = str(list_latent[i])
                map_lin1_out[key] = lin_out[0].detach().numpy()
                map_lin2_out[key] = lin_out[1].detach().numpy()
                map_lin3_out[key] = lin_out[2].detach().numpy()
        list_idx = np.array(list_of_idx)
        list_latent = np.array(list_latent)
        list_mean = np.array(list_mean)
        list_std = np.array(list_std)
        if running_experiment == '1':
            self.experiement_one_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == '2':
            self.experiment_two_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == '3':
            self.experiment_three_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == '4':
            self.experiment_four_plot_helper(
                list_idx, list_latent, list_mean, list_std, dim_manipulate, new_dim_value, decoder_output)
        elif running_experiment == 'a':
            self.experiment_a_plot_helper(
                map_lin1_out, map_lin2_out, map_lin3_out)
        elif running_experiment == '5':
            self.experiment_five_plot_helper(
                first_shape_list_of_idx, first_shape_list_latent, second_shape_list_of_idx, second_shape_list_latent)
        elif running_experiment == 'b':
            self.experiment_b_save_linear_layer_outputs(map_lin1_out)
        elif running_experiment == 'c':
            print("Investigating neuron behaviors for X =", x_value)
            self.experiment_c_x_val_neuron_fired(x_value)
        elif running_experiment == 'd':
            print("Investigating neuron behaviors for neuron index =", neuron_idx)
            print("Across all X values")
            self.experiment_d_neuron_idx_fired_for_all_x(neuron_idx)
        elif running_experiment == 'e':
            print("Investigating neuron behaviors for X =", x_value)
            print("Using z-score")
            self.experiment_e_neuron_fired_z_score(x_value)
        elif running_experiment == 'f':
            print("Investigating neuron behaviors for neuron index =", neuron_idx)
            print("Using z-score")
            self.experiment_f_neuron_idx_fired_for_all_x_z_score(neuron_idx)
        elif running_experiment == 'g':
            print("Investigating neuron behaviors for neuron index =", neuron_idx, "and X =", x_value)
            print("Calculating Bayesian Probabilities")
            self.experiment_g_neuron_idx_x_val_bayes(neuron_idx, x_value)
        elif running_experiment == 'h':
            print("Investigating neuron behaviors for neuron index =", neuron_idx)
            print("Calculating Bayesian Probabilities")
            self.experiment_h(neuron_idx)
        