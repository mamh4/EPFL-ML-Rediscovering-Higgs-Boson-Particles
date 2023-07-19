from matplotlib import pyplot as plt

def plot_features():
    # for f in x.columns:
    # plt.scatter(np.arange(len(train_data[f])), train_data[f],c = prediction_numeric, cmap='rainbow')
    # plt.title(f)
    # plt.xlabel('')
    # plt.grid()
    # plt.show()
    # Determine the number of columns in the grid (e.g., 2 columns for 2 plots side by side)
    num_columns = 5

    # Calculate the number of rows needed to accommodate all the plots
    num_rows = len(x.columns) // num_columns + 1

    # Increase the size of each subplot
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 12))

    # Flatten the axs array so that we can iterate over all subplots using a single loop
    axs = axs.flatten()

    for i, f in enumerate(x.columns):
        # Convert the 'Prediction' column to numeric values
        prediction_numeric, _ = pd.factorize(train_data['Prediction'])

        # Plot the scatter plot in the current subplot
        axs[i].scatter(np.arange(len(train_data[f])), train_data[f], c=prediction_numeric, cmap='rainbow')
        axs[i].set_title(f)
        axs[i].set_xlabel('')
        axs[i].grid()

    # Hide any empty subplots
    for j in range(len(x.columns), len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()