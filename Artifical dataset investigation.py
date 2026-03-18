# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:10:37 2026

@author: mikey

We are investigating the topology of a neural network as it groks

We use a theoretical dataset (a^c+b^c) mod p = output as our dataset for training.
We want to measure the difference in the neural network from the output.
"""


import pandas as pd
import numpy as np
# ============================================================================
# Functions
def dataframeToNumpy(df,key):
    return 

# ============================================================================
# MAIN
# ============================================================================

# should effectively be a ring
# x + 0 mod p = z 
# Define parameters
p=113
cutoff = 120

# Create an empty DataFrame with specific columns
df = pd.DataFrame(columns=["x", "y", "z"])
print(df)
print("Is DataFrame empty?", df.empty)
for a in range(1,cutoff):
    new_row = {"x" :a ,"y" :0 , "z": a%p}
    df.loc[len(df)] = new_row
print(df)

x_array = df['x'].to_numpy()
print(x_array)

# Plot
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

z = df['z'].to_numpy()
x = df['x'].to_numpy()
y = df['y'].to_numpy()

# Create scatter plot
ax.scatter3D(x, y, z, color='red', marker='o')

# Labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatterplot')

plt.show()


# ============================================================================
# two variables
# ============================================================================

# x + y mod p = z 
# Define parameters
p=113
cutoff = 120

# Create an empty DataFrame with specific columns
df = pd.DataFrame(columns=["x", "y", "z"])
print(df)
print("Is DataFrame empty?", df.empty)
for b in range(1,cutoff): 
    for a in range(1,cutoff):
        new_row = {"x" :a ,"y" :b , "z": (a+b)%p}
        df.loc[len(df)] = new_row
print(df)

x_array = df['x'].to_numpy()
print(x_array)

# Plot
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

z = df['z'].to_numpy()
x = df['x'].to_numpy()
y = df['y'].to_numpy()

# Create scatter plot
ax.scatter3D(x, y, z, color='red', marker='o')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatterplot')

plt.show()
