#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

columns = ["Farrah", "Fred", "Felicia"]
rows = ["apples", "bananas", "oranges", "peaches"]
colors = {"apples": "red", "bananas": "yellow", "oranges": "#ff8000", "peaches": "#ffe5b4"}

bar_width = 0.5
bar_positions = np.arange(len(columns))

for i, row in enumerate(rows):
    plt.bar(
        bar_positions,
        fruit[i, :],
        width=bar_width,
        bottom=np.sum(fruit[:i, :], axis=0),
        label=row,
        color=colors[row.lower()]  # Use lower() to handle case insensitivity
    )

#plt.xlabel('Person')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.xticks(bar_positions, columns)
plt.yticks(np.arange(0, 81, 10))
plt.legend()
plt.show()
