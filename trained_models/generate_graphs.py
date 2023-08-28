import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class GenerateGraph():
    
    def __init__(self) -> None:
        pass

    
    def histogram(self, data, title, x_label, y_label, bins):
        plt.hist(data, bins = bins)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    
    def line_plot(self, data, title, x_label, y_label):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    
    def epoch_line_plot(self, file_path):
        # Load training history from the CSV file
        history_df = pd.read_csv(file_path)

        # Extract data from the history DataFrame
        epochs = []

        for i in range(len(history_df)):
            epochs.append(i + 1)

        loss = history_df['loss']
        accuracy = history_df['accuracy']
        val_loss = history_df['val_loss']
        val_accuracy = history_df['val_accuracy']

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot loss
        ax1.plot(epochs, loss, label='Training Loss')
        ax1.plot(epochs, val_loss, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(epochs, accuracy, label='Training Accuracy')
        ax2.plot(epochs, val_accuracy, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()


    def cm_heatmap(self, file_path, save_path):
        # Load confusion matrix from CSV file
        confusion_df = pd.read_csv(file_path, index_col=0)

        # Convert the DataFrame to a numpy array
        confusion_matrix = confusion_df.values

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, cmap='YlGnBu', interpolation='nearest')
        plt.title('Confusion Matrix Heatmap')
        plt.colorbar()

        # Set x and y labels
        plt.xticks(np.arange(len(confusion_df.index)), confusion_df.index, rotation=45, ha='right')
        plt.yticks(np.arange(len(confusion_df.index)), confusion_df.index)

        # Display values in the cells
        for i in range(len(confusion_df.index)):
            for j in range(len(confusion_df.columns)):
                plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

        # Adjust layout
        plt.tight_layout()

        # Show the heatmap
        #plt.show()
        plt.savefig(save_path, bbox_inches='tight')

    

    def plot_top_classes_histogram(self, confusion_matrix_df, top_k):
        # Convert the DataFrame to a numpy array
        confusion_matrix = confusion_matrix_df.values

        # Get the indices of classes with top K confusion values
        top_k_classes = np.argsort(np.sum(confusion_matrix, axis=1))[-top_k:]

        # Calculate the sum of confusion values for the top K classes
        top_k_confusion_sum = np.sum(confusion_matrix[top_k_classes], axis=1)

        # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(top_k), top_k_confusion_sum, tick_label=top_k_classes)
        plt.title(f'Confusion Sum Histogram for Top {top_k} Classes')
        plt.xlabel('Class Index')
        plt.ylabel('Confusion Sum')

        # Show the histogram
        plt.show()


    def plot_worst_classes_histogram(self, confusion_matrix_df, worst_k):
        # Convert the DataFrame to a numpy array
        confusion_matrix = confusion_matrix_df.values

        # Calculate sum of true positives for each class
        true_positives = np.diag(confusion_matrix)

        # Get the indices of classes with worst K confusion values
        worst_k_classes = np.argsort(true_positives)[:worst_k]

        # Calculate the sum of confusion values for the worst K classes
        worst_k_confusion_sum = np.sum(confusion_matrix[worst_k_classes], axis=1)

        # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(worst_k), worst_k_confusion_sum, tick_label=worst_k_classes)
        plt.title(f'Confusion Sum Histogram for Worst {worst_k} Classes')
        plt.xlabel('Class Index')
        plt.ylabel('Confusion Sum')

        # Show the histogram
        plt.show()


    def plot_top_classes_heatmap_with_labels(self, confusion_matrix_df, top_k):
        # Convert the DataFrame to a numpy array
        confusion_matrix = confusion_matrix_df.values

        # Get the indices of classes with top K confusion values
        top_k_classes = np.argsort(np.sum(confusion_matrix, axis=1))[-top_k:]

        # Extract the confusion matrix for top K classes
        confusion_matrix_top_k = confusion_matrix[top_k_classes][:, top_k_classes]

        # Create a heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(confusion_matrix_top_k, cmap='YlGnBu', interpolation='nearest')
        plt.title(f'Confusion Matrix Heatmap for Top {top_k} Classes')
        plt.colorbar()

        # Set x and y labels
        class_labels = confusion_matrix_df.index
        class_labels_top_k = class_labels[top_k_classes]
        plt.xticks(np.arange(top_k), class_labels_top_k, rotation=45, ha='right')
        plt.yticks(np.arange(top_k), class_labels_top_k, rotation=0, ha='right')

        # Display class indices as labels on the sides and values within the heatmap
        for i in range(top_k):
            for j in range(top_k):
                plt.text(j, i, str(confusion_matrix_top_k[i, j]), ha='center', va='center', color='black')

        # Display class indices as labels on the sides
        for i in range(top_k):
            plt.text(-0.5, i, str(top_k_classes[i]), ha='right', va='center', color='black')
            plt.text(i, -0.5, str(top_k_classes[i]), ha='center', va='top', color='black')

        # Adjust layout
        plt.tight_layout()

        # Show the heatmap
        plt.show()


    def plot_worst_classes_heatmap_with_labels_and_values(self, confusion_matrix_df, worst_k):
        # Convert the DataFrame to a numpy array
        confusion_matrix = confusion_matrix_df.values

        # Calculate sum of true positives for each class
        true_positives = np.diag(confusion_matrix)

        # Get the indices of classes with worst K confusion values
        worst_k_classes = np.argsort(true_positives)[:worst_k]

        # Extract the confusion matrix for worst K classes
        confusion_matrix_worst_k = confusion_matrix[worst_k_classes][:, worst_k_classes]

        # Create a heatmap
        plt.figure(figsize=(12, 10))
        heatmap = plt.imshow(confusion_matrix_worst_k, cmap='YlOrRd', interpolation='nearest')
        plt.title(f'Confusion Matrix Heatmap for Worst {worst_k} Classes')
        plt.colorbar(heatmap, fraction=0.045, pad=0.04)

        # Set x and y labels
        class_labels = confusion_matrix_df.index
        class_labels_worst_k = class_labels[worst_k_classes]
        plt.xticks(np.arange(worst_k), class_labels_worst_k, rotation=45, ha='right')
        plt.yticks(np.arange(worst_k), class_labels_worst_k, rotation=0, ha='right')

        # Display class indices as labels on the sides and values within the heatmap
        for i in range(worst_k):
            for j in range(worst_k):
                plt.text(j, i, str(confusion_matrix_worst_k[i, j]), ha='center', va='center', color='black')

        # Display class indices as labels on the sides
        for i in range(worst_k):
            plt.text(-0.5, i, str(worst_k_classes[i]), ha='right', va='center', color='black')
            plt.text(i, -0.5, str(worst_k_classes[i]), ha='center', va='top', color='black')

        # Adjust layout
        plt.tight_layout()

        # Show the heatmap
        plt.show()


def main():

    simple_confusion_matrix_path = 'simple_2023_08_20_18_18_10_confusion_matrix.csv'
    simple_epoch_history_path = 'simple_2023_08_20_18_18_10_history.csv'

    resnet_confusion_matrix_path = 'resnet_2023_08_23_14_05_52_confusion_matrix.csv'

    simple_confusion_df = pd.read_csv(simple_confusion_matrix_path, index_col=0)

    graph_plotter = GenerateGraph()
    #graph_plotter.epoch_line_plot(simple_epoch_history_path)
    #graph_plotter.plot_top_classes_histogram(simple_confusion_df, top_k=10)
    #graph_plotter.plot_worst_classes_histogram(simple_confusion_df, worst_k=10)
    #graph_plotter.cm_heatmap(simple_confusion_matrix_path, 'simple_confusion_matrix.png')
    graph_plotter.plot_top_classes_heatmap_with_labels(simple_confusion_df, top_k=10)
    graph_plotter.plot_worst_classes_heatmap_with_labels_and_values(simple_confusion_df, worst_k=10)


if __name__ == '__main__':
    main()
