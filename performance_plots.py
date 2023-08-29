import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

folder = "Rwanda"
approach = "seventh"
drop = 'none'
test_prop = 0.2

os.chdir(f'/Users/naiacasina/Library/CloudStorage/OneDrive-UCB-O365/SEM2/B2P/Data/')

with open(f'{folder}/Saved data/performance_{approach}_test_size_{test_prop}_dropped_{drop}.pkl', 'rb') as f:
    performance_df = pickle.load(f)

# ---- First performance plot ----
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 4))

# plot accuracy, recall, and precision
ax.plot(performance_df['Classifier'], performance_df['Accuracy'], marker='o', label='Accuracy')
ax.plot(performance_df['Classifier'], performance_df['Recall'], marker='o', label='Recall')
ax.plot(performance_df['Classifier'], performance_df['Precision'], marker='o', label='Precision')

# set plot labels and title
ax.set_xlabel('Methods')
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison of Methods')

# customize plot appearance
ax.legend()
ax.grid(True)

# set the y-limits
ax.set_ylim(0.8, 1)  # Adjust the values as needed

# save the figure with higher resolution (e.g., DPI = 300)
dpi = 300  # Adjust the DPI value as needed
plt.savefig(f'{folder}/ML/Performance plots/performance_plot_{approach}_test_size_{test_prop}_dropped.png', dpi=dpi)

# show the plot
plt.show()

# ----- First plot for recall ------
# list of test_prop values
test_props = [0.2, 0.3, 0.4, 0.5]
# metric
metric = 'Accuracy'

fig, ax = plt.subplots(figsize=(8, 4))

# iterate over test_prop values
for i, test_prop in enumerate(test_props):
    # filter the dataframe based on the test_prop value
    with open(
            f'{folder}/Saved data/performance_{approach}_test_size_{test_prop}.pkl',
            'rb') as f:
        performance_df = pickle.load(f)

    # define the pastel color for the line plot
    color = sns.color_palette('pastel')[i]

    # plot the recall values with pastel color
    ax.plot(performance_df['Classifier'], performance_df[metric], marker='o', label=f'Test_prop={test_prop}', color=color)

# set plot labels and title
ax.set_ylabel(f'{metric}')
ax.set_title(f'{metric} Comparison for Different Test Proportions')

# customize plot appearance
ax.legend()
ax.grid(True)

# set the y-limits
ax.set_ylim(0.55, 0.9)  # Adjust the values as needed

# save the figure
plt.savefig(f'{folder}/ML/Performance plots/{metric}_plot_{approach}.png')

# show the plot
plt.show()

# ---- Second performance plot ----
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# define a color palette for the classifiers
colors = ['#8fbdd8', '#f7ae76', '#a8d8b9', '#f392bd', '#cbb0e3']

# accuracy
plt.subplot(1, 3, 1)
sns.barplot(x='Accuracy', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Accuracy')
plt.ylabel('Classifier')
plt.xlim(0, 1)

# recall
plt.subplot(1, 3, 2)
sns.barplot(x='Recall', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Recall')
plt.ylabel('')
plt.xlim(0, 1)

# precision
plt.subplot(1, 3, 3)
sns.barplot(x='Precision', y='Classifier', data=performance_df, palette=colors)
plt.xlabel('Precision')
plt.ylabel('')
plt.xlim(0, 1)

plt.tight_layout()
# save the figure
plt.savefig(f'{folder}/ML/Performance plots/performance_plot2_{approach}_test_size_{test_prop}.png')

plt.show()


