import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
block_data = pd.read_csv('BLOCK_width_timing_data.csv')
tile_data = pd.read_csv('tile_width_timing_data.csv')

# Prepare the plot
plt.figure(figsize=(12, 8))

# Plot data from the first CSV (BlockWidth)
for block_width in block_data['BlockWidth'].unique():
    subset = block_data[block_data['BlockWidth'] == block_width]
    plt.errorbar(subset['MatrixSize'], subset['GPUAverage'], yerr=subset['GPUError'], fmt='-o', label=f'Block Width {block_width}')

# Plot data from the second CSV (TileWidth)
for tile_width in tile_data['TileWidth'].unique():
    subset = tile_data[tile_data['TileWidth'] == tile_width]
    plt.errorbar(subset['MatrixSize'], subset['GPUAverage'], yerr=subset['GPUError'], fmt='--^', label=f'Tile Width {tile_width}')

# Configure the plot
plt.title('Comparison of GPU Average Time vs. Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
