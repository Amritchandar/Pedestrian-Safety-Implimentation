import logging

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("start...")
def parse_result_file(folder_path):
    """
    Parse the result .txt file to extract Predicted Boxes, Scores, and Labels.
    """
    # Initialize containers
    predicted_boxes = []
    predicted_scores = []
    predicted_labels = []

    logger.info("Parsing result...")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()


            # Parse sections
            current_section = None
            for line in lines:
                line = line.strip()
                if line.strip().startswith("device='cuda:0')") or line.strip().startswith("\n") or line.strip()=='':
                    continue
                if line.startswith("Predicted Boxes:"):
                    current_section = "boxes"
                elif line.startswith("Predicted Scores:"):
                    current_section = "scores"
                elif line.startswith("Predicted Labels:"):
                    current_section = "labels"

                elif current_section == "boxes":
                    # Parse Predicted Boxes
                    box_line = None
                    if line.startswith("tensor"):
                        box_line = line[line.find('[')+2:line.find(']')]
                    elif line.startswith("["):
                        box_line = line[line.find('[')+1:line.find(']')]
                    if box_line:
                        box_values = [float(x.strip()) for x in box_line.split(",")]
                        predicted_boxes.append(box_values[:3])
                elif current_section == "scores":
                    # Parse Predicted Scores
                    score_line = None
                    if line.startswith("tensor"):
                        score_line = line[line.find('[')+2:line.find(']')]
                    elif len(line) > 0 and line.strip()[0].isdigit():
                        bracket_index = line.find(']')
                        comma_index = line.rfind(',')
                        score_line = line[0:bracket_index if bracket_index != -1 else comma_index]
                    if score_line:
                        score_values = [float(x.strip()) for x in score_line.split(",")]
                        predicted_scores += score_values
                elif current_section == "labels":
                    # Parse Predicted Labels
                    label_line = None
                    if line.startswith("tensor"):
                        label_line = line[line.find('[')+1:line.rfind(']')]
                    elif line.strip()[0].isdigit():
                        bracket_index = line.find(']')
                        comma_index = line.rfind(',')
                        label_line = line[0:bracket_index if bracket_index != -1 else comma_index]
                    if label_line:
                        label_values = [int(x.strip()) for x in label_line.split(",")]
                        predicted_labels += label_values

    logger.info("parse completed")
    # Convert to np array
    predicted_boxes = np.array(predicted_boxes)
    predicted_scores = np.array(predicted_scores)
    predicted_labels = np.array(predicted_labels)

    # indices = [i for i, subarray in enumerate(predicted_boxes) if len(subarray) == 7]
    # predicted_boxes = np.array(predicted_boxes, dtype=object)  # Ensure compatibility with varying inner sizes
    # predicted_scores = np.array(predicted_scores)
    # predicted_labels = np.array(predicted_labels)

    return predicted_boxes, predicted_scores, predicted_labels


def plot_3d_contour(predicted_boxes, predicted_scores, predicted_labels):
    # Filter data by label and plot
    fig = plt.figure(figsize=(18, 6))
    labels = [0, 1, 2]
    for i, label in enumerate(labels):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Filter points for the current label
        indices = np.where(predicted_labels == label)[0]
        x, y, z = predicted_boxes[indices, 0], predicted_boxes[indices, 1], predicted_boxes[indices, 2]
        scores = predicted_scores[indices]

        # Plot contours for the current label
        plot_3d_contour_for_label(label, x, y, z, scores, ax)

    plt.tight_layout()
    plt.show()


def plot_3d_contour_for_label(label, x, y, z, scores, ax):
    # Create a 3D grid
    grid_x, grid_y, grid_z = np.mgrid[
                             np.min(x):np.max(x):100j,  # 100 points along x
                             np.min(y):np.max(y):100j,  # 100 points along y
                             np.min(z):np.max(z):100j   # 100 points along z
                             ]

    # Interpolate scores onto the grid
    grid_scores = griddata(
        points=(x, y, z),
        values=scores,
        xi=(grid_x, grid_y, grid_z),
        method='linear'
    )

    # Contour levels
    levels = np.linspace(np.nanmin(grid_scores), np.nanmax(grid_scores), 10)

    sc = ax.scatter(x, y, z, c=scores, cmap='viridis', s=50)

    # Add colorbar
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Score')

    # Add labels
    ax.set_title('Scores contour for {}'.format(label))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # # Draw the contour plot (slice at mid-Z)
    # ax.contour3D(
    #     grid_x[:, :, 0],
    #     grid_y[:, :, 0],
    #     grid_scores[:, :, grid_scores.shape[2] // 2],
    #     levels=levels,
    #     cmap='viridis'
    # )
    #
    # # Set titles and labels
    # ax.set_title(f'3D Contour for Label {label}')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Score')


def plot_distance_vs_score(distance, scores, labels, name):
    fig, ax = plt.subplots(1, 3, figsize=(9, 5))

    for i, label in enumerate([0, 1, 2]):

        # Filter points for the current label
        indices = np.where(labels == label)[0]
        x = distance[indices]
        y = scores[indices]
        stats = {
            "count": len(y),
            "mean": float(f"{np.mean(y):.3f}"),
            "std": float(f"{np.std(y):.3f}"),  # Standard deviation
            "min": float(f"{np.min(y):.3f}"),
            "25%": float(f"{np.percentile(y, 25):.3f}"),  # 25th percentile
            "50%": float(f"{np.median(y):.3f}"),  # Median
            "75%": float(f"{np.percentile(y, 75):.3f}"),  # 75th percentile
            "max": float(f"{np.max(y):.3f}")
        }
        # ax[i].hexbin(x, y, gridsize=50, cmap='viridis', reduce_C_function=np.mean, label = f'Dist vs score for {i} label')
        ax[i].scatter(x, y)
        ax[i].set_xlabel('distance')
        ax[i].set_ylabel('score')
        ax[i].axhline(float(stats["75%"]), color='red', linestyle='--', label=f'75%ile = {stats["75%"]}')
        ax[i].axhline(float(stats["50%"]), color='green', linestyle='--', label=f'Median = {stats["50%"]}')
        ax[i].set_title(f'label {i}')
        ax[i].legend()

    #plt.title(f"Distance vs Score for {name}")
    plt.tight_layout()
    plt.show()


def plot_dist_quantile_plot(distance):
    stats = {
        "count": len(distance),
        "mean": float(f"{np.mean(distance):.3f}"),
        "std": float(f"{np.std(distance):.3f}"),  # Standard deviation
        "min": float(f"{np.min(distance):.3f}"),
        "25%": float(f"{np.percentile(distance, 25):.3f}"),  # 25th percentile
        "50%": float(f"{np.median(distance):.3f}"),  # Median
        "75%": float(f"{np.percentile(distance, 75):.3f}"),  # 75th percentile
        "max": float(f"{np.max(distance):.3f}")
    }
    # Plot the quantile plot
    plt.figure(figsize=(8, 6))
    plt.boxplot(distance, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Quantile Plot")
    plt.xlabel("Quantiles (Cumulative Probability)")
    plt.ylabel("Data Values")
    plt.axvline(float(stats["mean"]), color='red', linestyle='--', label=f'Mean = {stats["mean"]}')
    plt.axvline(float(stats["50%"]), color='green', linestyle='--', label=f'Median = {stats["50%"]}')
    plt.legend()
    plt.grid()
    plt.show()

    logger.info(f"distance stats: {stats}")

def threshold_cut(boxes, scores, labels, threshold):
    indices = np.where(scores>=threshold)
    return boxes[indices], scores[indices], labels[indices]



def calculate_dist(row):
    return np.sqrt(row[0]**2+row[1]**2+row[2]**2)


def process_model_scores(distances, scores, labels, interval_size=0.1, sigma_threshold=6):
    """
    Process the distances and scores into grouped intervals and remove outliers.
    """
    unique_labels = np.unique(labels)
    processed_distances = []
    processed_scores = []
    processed_labels = []

    for label in unique_labels:
        # Filter data for the current label
        mask = labels == label
        dist = distances[mask]
        score = scores[mask]

        # Sort by distance
        sorted_indices = np.argsort(dist)
        dist = dist[sorted_indices]
        score = score[sorted_indices]

        # Create intervals
        min_dist = np.min(dist)
        max_dist = np.max(dist)
        bins = np.arange(min_dist, max_dist + interval_size, interval_size)

        for i in range(len(bins) - 1):
            # Define the interval
            start = bins[i]
            end = bins[i + 1]

            # Select scores in the interval
            in_interval = (dist >= start) & (dist < end)
            scores_in_interval = score[in_interval]

            if len(scores_in_interval) > 0:
                # Compute mean and standard deviation
                mean_score = np.mean(scores_in_interval)
                std_score = np.std(scores_in_interval)

                # Remove outliers
                valid_scores = scores_in_interval[
                    np.abs(scores_in_interval - mean_score) <= sigma_threshold * std_score
                    ]

                if len(valid_scores) > 0:
                    # Append results
                    processed_distances.append((start + end) / 2)  # Midpoint of the interval
                    processed_scores.append(np.mean(valid_scores))
                    processed_labels.append(label)

    return np.array(processed_distances), np.array(processed_scores), np.array(processed_labels)


def process_model_with_quantile(distances, scores, labels, model_name, interval_size=0.3):
    bins = np.arange(min(distances), max(distances) + interval_size, interval_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = ['blue', 'green', 'orange']
    for label in [0, 1, 2]:
        # Filter data for the current label
        mask = labels == label
        dist = distances[mask]
        score = scores[mask]

        # Sort by distance
        sorted_indices = np.argsort(dist)
        dist = dist[sorted_indices]
        score = score[sorted_indices]

        bin_bucket = []

        for i in range(len(bins) - 1):
            # Define the interval
            start = bins[i]
            end = bins[i + 1]

            # Select scores in the interval
            in_interval = (dist >= start) & (dist < end)
            scores_in_bin = score[in_interval]

            if len(scores_in_bin) > 0:
                bin_bucket.append({
                    'median': float(f"{np.median(scores_in_bin):.3f}"),
                    'q25': float(f"{np.percentile(scores_in_bin, 25):.3f}"),
                    'q75': float(f"{np.percentile(scores_in_bin, 75):.3f}"),
                    'min': float(f"{np.min(scores_in_bin):.3f}"),
                    'max': float(f"{np.max(scores_in_bin):.3f}"),
                    'bin_center': float(f"{(bins[i] + bins[i + 1]) / 2:.3f}")
                })
            else:
                bin_bucket.append(None)

        data = replace_none_with_neighbors(bin_bucket)

        bin_centers = [stat['bin_center'] for stat in data]
        medians = [stat['median'] for stat in data]
        q25s = [stat['q25'] for stat in data]
        q75s = [stat['q75'] for stat in data]
        mins = [stat['min'] for stat in data]
        maxs = [stat['max'] for stat in data]

        # Plot median line
        axes[label].plot(bin_centers, medians, label=f'Median', color=colors[label], lw=2)

        # Fill area between q25 and q75
        axes[label].fill_between(bin_centers, q25s, q75s, color=colors[label], alpha=0.2, label=f'Q25-Q75')

        # Optionally plot min-max as a light outline
        axes[label].fill_between(bin_centers, mins, maxs, color=colors[label], alpha=0.1, label=f'Min-Max')

        # Set subplot title and labels
        axes[label].set_title(f"Statistics for {label}")
        axes[label].set_xlabel('Distance')
        axes[label].set_ylabel('Score')
        axes[label].legend()
        axes[label].grid(True)

    plt.tight_layout()
    plt.show()


def replace_none_with_neighbors(lst):
    result = lst.copy()  # Create a copy of the list

    for i in range(len(result)):
        if result[i] is None:
            prev_value = None
            next_value = None

            # Find the previous non-None value
            for j in range(i - 1, -1, -1):
                if result[j] is not None:
                    prev_value = result[j]
                    break

            # Find the next non-None value
            for j in range(i + 1, len(result)):
                if result[j] is not None:
                    next_value = result[j]
                    break

            # Replace None with the appropriate value
            if prev_value is not None and next_value is not None:
                result[i] = {
                    'median': (prev_value['median']+next_value['median'])/2,
                    'q25': (prev_value['q25']+next_value['q25'])/2,
                    'q75': (prev_value['q75']+next_value['q75'])/2,
                    'min': (prev_value['min']+next_value['min'])/2,
                    'max': (prev_value['max']+next_value['max'])/2,
                    'bin_center': (prev_value['bin_center']+next_value['bin_center'])/2
                }
            elif prev_value is not None:
                result[i] = {
                    'median': prev_value['median'],
                    'q25': prev_value['q25'],
                    'q75': prev_value['q75'],
                    'min': prev_value['min'],
                    'max': prev_value['max'],
                    'bin_center': prev_value['bin_center']
                }
            elif next_value is not None:
                result[i] = {
                    'median': next_value['median'],
                    'q25': next_value['q25'],
                    'q75': next_value['q75'],
                    'min': next_value['min'],
                    'max': next_value['max'],
                    'bin_center': next_value['bin_center']
                }
            else:
                result[i] = np.nan  # Fallback if the list is entirely None

    return result



if __name__ == '__main__':

    """
    Process the .txt file to group results by label.
    """
    # Parse the .txt file
    # predicted_boxes, predicted_scores, predicted_labels = parse_result_file("demo_result")
    # # predicted_boxes, predicted_scores, predicted_labels = threshold_cut(predicted_boxes, predicted_scores, predicted_labels, threshold=0.5)
    # box_distance = np.apply_along_axis(calculate_dist, 1, predicted_boxes)
    # plot_distance_vs_score(box_distance, predicted_scores, predicted_labels)
    # plot_dist_quantile_plot(box_distance)

    box_pp, score_pp, label_pp = parse_result_file("demo_result")
    box_a2, score_a2, label_a2 = parse_result_file("demo_result_parta2")
    dist_pp = np.apply_along_axis(calculate_dist, 1, box_pp)
    dist_a2 = np.apply_along_axis(calculate_dist, 1, box_a2)
    # process_model_with_quantile(dist_pp, score_pp, label_pp, "pointpillar")
    # process_model_with_quantile(dist_a2, score_a2, label_a2, "part_a2")
    # process_model_with_quantile(comb_dist, comb_score, comb_label, "MOE")
    # plot_3d_contour(box_pp, score_pp, label_pp)
    # plot_3d_contour(box_a2, score_a2, label_a2)
    # prop_dist_pp, prop_score_pp, prop_label_pp = process_model_scores(dist_pp, score_pp, label_pp)
    # prop_dist_a2, prop_score_a2, prop_label_a2 = process_model_scores(dist_a2, score_a2, label_a2)
    # comb_dist, comb_score, comb_label, comb_best_model = combine_models_with_threshold(prop_dist_pp, prop_score_pp, prop_label_pp, prop_dist_a2, prop_score_a2, prop_label_a2)
    # plot_distance_vs_score(prop_dist_pp, prop_score_pp, prop_label_pp, 'PointPillar')
    # plot_distance_vs_score(prop_dist_a2, prop_score_a2, prop_label_a2, 'Part A2')
    # plot_distance_vs_score(comb_dist, comb_score, comb_label, 'MOE')
    # plot_dist_quantile_plot(dist_pp)
    # plot_dist_quantile_plot(dist_a2)
    # plot_dist_quantile_plot(comb_dist)


