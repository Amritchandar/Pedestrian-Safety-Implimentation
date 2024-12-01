
import json
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import statistics

label_dict = {0:[],
              1:[],
              2:[]
              }


directory = 'parta2_demo_result'
file_list = os.listdir(directory)

if os.path.exists('output.log'):
    os.remove("output.log")

def calc_statistics(data):
    mean = statistics.mean(data)
    mode = statistics.mode(data)  # Most common value
    std_dev = statistics.stdev(data)  # Standard deviation
    minimum = min(data)  # Minimum value
    maximum = max(data)  # Maximum value

    # Display the results
    print(f"Mean: {mean}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")

def graph_dict():

    data = [label_dict[0],label_dict[1],label_dict[2]]
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, patch_artist=True, notch=True, showmeans=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    meanprops=dict(marker='o', markeredgecolor='red', markerfacecolor='red'))

    # Add labels and title
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Cars', 'Pedestrians', 'Bikes'])
    ax.set_title('BoxPlot of PartA2 Scores')
    ax.set_ylabel('Scores')

    # Compute statistics and annotate each boxplot
    for i, dataset in enumerate(data):
        q1 = np.percentile(dataset, 25)  # First quartile (Q1)
        median = np.median(dataset)  # Median
        q3 = np.percentile(dataset, 75)  # Third quartile (Q3)
        iqr = q3 - q1  # Interquartile range

        # Calculate lower and upper whiskers within the IQR range
        lower_whisker = np.min([j for j in dataset if j >= (q1 - 1.5 * iqr)])
        upper_whisker = np.max([j for j in dataset if j <= (q3 + 1.5 * iqr)])

        mean = np.mean(dataset)  # Mean
  
        # Annotate key statistics above their positions
        ax.text(i + 1.35, mean, f"Mean: {mean:.2f}", color='red', ha='center', fontsize=12)
        ax.text(i + 1.35, median, f"Median: {median:.2f}", color='blue', ha='center', fontsize=12)
        ax.text(i + 1.35, q1, f"Q1: {q1:.2f}", color='green', ha='center', fontsize=12)
        ax.text(i + .7, q3, f"Q3: {q3:.2f}", color='green', ha='center', fontsize=12)
        ax.text(i + 1.35, lower_whisker, f"Min: {lower_whisker:.2f}", color='purple', ha='center', fontsize=12)
        ax.text(i + 1.35, upper_whisker, f"Max: {upper_whisker:.2f}", color='purple', ha='center', fontsize=12)

    plt.tight_layout()

    

def read_tensors(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    
    # Extract the Predicted Scores tensor
    scores_pattern = r"Predicted Scores:\stensor\((\[.*?\])"
    scores_match = re.search(scores_pattern, data, re.DOTALL)
    if scores_match:
        scores_str = scores_match.group(1)
        scores = np.array(eval(scores_str))
    else:
        scores = np.array([])

    # Extract the Predicted Labels tensor
    labels_pattern = r"Predicted Labels:\stensor\((\[.*?\])"
    labels_match = re.search(labels_pattern, data, re.DOTALL)
    if labels_match:
        labels_str = labels_match.group(1)
        labels = np.array(eval(labels_str))
    else:
        labels = np.array([])

    return labels, scores


for i in file_list:
    predicted_labels, predicted_scores = read_tensors(directory+"/"+str(i))
    for index, value in enumerate(predicted_scores):
        if(predicted_labels[index]==0 and value>0):
            label_dict[predicted_labels[index]].append(value)
        elif(predicted_labels[index]==1 and value>0):
            label_dict[predicted_labels[index]].append(value)
        elif(predicted_labels[index]==2 and value>0):
            label_dict[predicted_labels[index]].append(value)



graph_dict()
plt.show()

print(f"number of cars: {len(label_dict[0])}")
print(f"number of pedestrians: {len(label_dict[1])}")
print(f"number of bikes: {len(label_dict[2])}")
# print("statistics for cars:")
# calc_statistics(label_dict[0])
# print("statistics for pedestrian:")
# calc_statistics(label_dict[1])
# print("statistics for bike:")
# calc_statistics(label_dict[2])

with open('output.log','a') as output_file:
    json.dump(label_dict,output_file,indent=4)

