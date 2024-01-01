import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = 'communication_cost.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Extracting necessary columns for the scatter plot
x = data['Clinet Communication overhead']
y = data['Acc']
methods = data['Methods']

# Define different markers and a color map for each method
markers = ['o', 's', '^', 'D', 'p', '*', 'x', '+', 'v', '<', '>', '1', '2', '3', '4', 'h']
colors = plt.cm.get_cmap('tab20', len(methods))
method_colors = ['red' if method == 'FedPPN' else colors(i) for i, method in enumerate(methods)]

# Creating the scatter plot
plt.figure(figsize=(12, 8))
plt.grid(True, zorder=0)
# Plotting each method with a unique marker and color
for i, method in enumerate(methods):
    marker = markers[i % len(markers)]  # Cycle through markers if there are more methods than markers
    plt.scatter(x[i], y[i], marker=marker, color=method_colors[i], s=150 if method == 'FedPPN' else 60, zorder=3)

# Manually adjusting annotations to prevent overlap for specific methods
for i, txt in enumerate(methods):
    offset_x = 0
    offset_y = 10
    if txt == "Local":
        offset_y = -5
        offset_x = 27
    elif txt == 'FedProto':
        offset_y = -5
        offset_x = 40
    elif txt == 'FedGH':
        offset_y = -5
        offset_x = 30
    elif txt == 'FedPPN':
        offset_y = -5
        offset_x = 35

    plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=14)

# Adding labels and title
plt.xlabel('Average communication cost of clients (MB) ',fontsize=14)
plt.ylabel('Accuracy',fontsize=14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

# plt.title('Comparison of ACC and Communication Overhead for Different Methods')

# Adjusting legend to include unique markers with matching colors for each method
legend_elements = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor=method_colors[i],
                              markersize=15 if txt == 'FedPPN' else 10) for i, txt in enumerate(methods)]
plt.legend(handles=legend_elements, labels=methods.tolist())

# Showing the plot with a grid
save_path='communication_cost.png'
plt.savefig(save_path, dpi=300)
plt.show()



