from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
import seaborn as sns
from scipy.stats import norm


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
            sample_zCx = self.model.reparameterize(mean, log_var)

            output, lin_outputs, conv_outputs, conv_weights_tuple = self.model.decoder(sample_zCx)

        return sample_zCx, params_zCx, output, mean, log_var, lin_outputs, conv_outputs, conv_weights_tuple

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

    def experiment_a_plot_helper(self, list_of_idx, list_latent, list_lin1_out, list_lin2_out, list_lin3_out):
        # get the mean and std of the each neuron in each linear layer
        lin1_means, lin1_stds, lin2_means, lin2_stds, lin3_means, lin3_stds = self.experiment_a_linear_layer_mean_std(list_lin1_out, list_lin2_out, list_lin3_out)
        # find the neurons that fire bigger than the mean + 1 std
        lin1_neurons_fired = self.experiment_a_find_neurons_fired(list_lin1_out, lin1_means, lin1_stds)
        lin2_neurons_fired = self.experiment_a_find_neurons_fired(list_lin2_out, lin2_means, lin2_stds)
        lin3_neurons_fired = self.experiment_a_find_neurons_fired(list_lin3_out, lin3_means, lin3_stds)
        print("lin1", lin1_neurons_fired)
        print("lin2", lin2_neurons_fired)
        print("lin3", lin3_neurons_fired)
        # plot the activation of neurons for each experiment and have the neuron firing in a gradient of colors

    def experiment_a_find_neurons_fired(self, list_lin_out, lin_means, lin_stds):
        # have neurons_fired in an array per experiment, (experiment_size, num_neurons_fired)
        neurons_fired = [[] for i in range(list_lin_out.shape[0])]
        # number of experiments
        for i in range(list_lin_out.shape[0]):
            # number of neurons
            for j in range(list_lin_out.shape[2]):
                if list_lin_out[i, :, j] > lin_means[:, j] + 2 * lin_stds[:, j]:
                    # append the index of the neuron 
                    neurons_fired[i].append(j)
        return neurons_fired

    def experiment_a_linear_layer_mean_std(self, 
                                           list_lin1_out, # experiment_size, 1, 256
                                           list_lin2_out, # experiment_size, 1, 256
                                           list_lin3_out): # experiment_size, 1, 512
        lin1_means = np.mean(list_lin1_out, axis=0)
        lin1_stds = np.std(list_lin1_out, axis=0)
        lin2_means = np.mean(list_lin2_out, axis=0)
        lin2_stds = np.std(list_lin2_out, axis=0)
        lin3_means = np.mean(list_lin3_out, axis=0)
        lin3_stds = np.std(list_lin3_out, axis=0)
        return lin1_means, lin1_stds, lin2_means, lin2_stds, lin3_means, lin3_stds

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

    def main_experiment(self):

        list_of_idx = []
        list_latent = []
        list_mean = []
        list_std = []
        list_lin1_out = []
        list_lin2_out = []
        list_lin3_out = []

        running_experiment = False
        while True:
            prompt = " Enter the experiment number, else enter N \n" + \
                "(Experiment 1): Incremently increase the x from 0 to 31, while measuring the difference between the \n" +  \
                "latent dimensions for each x when y = 0 v.s. y = 31 for the shape ellipse, rotation 0, and size 3?\n" + \
                "(Experiment 2): Hold X to be constant at 0, and incremently increase the y from 0 to 31, while visualizing the difference in latent space, for the shape ellipse, rotation 0, and size 3 \n" + \
                "(Experiment 3): Repetition of Experiment 2, except you can control shape, and input multiple possible X\n" +\
                "(Experiment a): Hold shape, scale, rotation, and x position constant, and only vary y position from 0 to 31, while measuring the mean and std of neurons in the three linear layers\n" + \
                "Experiment Number (ENTER): "

            automatic_choice = input(
                prompt).strip()
            if automatic_choice == '1':
                list_of_idx, list_latent = self.experiment_one_vary_top_bottom_y_for_all_x()
                running_experiment = '1'
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
                break
            elif automatic_choice == 'a':
                print("experiment a ")
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
                    # calling other experiment's helper to get the list of idx and list of latents
                    cur_list_of_idx, cur_list_latent = self.experiment_three_hold_x_constant_incremently_increase_y(
                        x_value, shape)
                    list_of_idx.extend(cur_list_of_idx)
                    list_latent.extend(cur_list_latent)
                running_experiment = 'a'
                break

            print("automatic choice", automatic_choice)

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

            samples_zCx, params_zCx, decoder_output, mean, log_var, lin_out, conv_out, conv_weights_tuple = self._compute_q_zCx_single(
                self.dataloader, list_of_idx[i])

            std = np.array(np.sqrt(np.exp(log_var[0])))
            mean = np.array(mean[0])

            list_mean.append(mean)
            list_std.append(std)
            list_lin1_out.append(lin_out[0])
            list_lin2_out.append(lin_out[1])
            list_lin3_out.append(lin_out[2])

        list_idx = np.array(list_of_idx)
        list_latent = np.array(list_latent)
        list_mean = np.array(list_mean)
        list_std = np.array(list_std)
        list_lin1_out = np.array(list_lin1_out)
        list_lin2_out = np.array(list_lin2_out)
        list_lin3_out = np.array(list_lin3_out)

        if running_experiment == '1':
            self.experiement_one_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == '2':
            self.experiment_two_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == '3':
            self.experiment_three_plot_helper(
                list_idx, list_latent, list_mean, list_std)
        elif running_experiment == 'a':
            self.experiment_a_plot_helper(
                list_idx, list_latent, list_lin1_out, list_lin2_out, list_lin3_out)