import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
block_data = pd.read_csv('BLOCK_width_timing_data.csv')
tile_data = pd.read_csv('tile_width_timing_data.csv')

# Specified widths to compare
widths_to_compare = [16, 25, 32]

# Prepare the plot
plt.figure(figsize=(10, 6))

for width in widths_to_compare:
    # Filter data for BlockWidth
    block_subset = block_data[block_data['BlockWidth'] == width]
    if not block_subset.empty:
        plt.errorbar(block_subset['MatrixSize'], block_subset['GPUAverage'], yerr=block_subset['GPUError'], fmt='-o', label=f'Block Width {width}')

    # Filter data for TileWidth
    tile_subset = tile_data[tile_data['TileWidth'] == width]
    if not tile_subset.empty:
        plt.errorbar(tile_subset['MatrixSize'], tile_subset['GPUAverage'], yerr=tile_subset['GPUError'], fmt='--^', label=f'Tile Width {width}')

# Configure the plot
plt.title('GPU Average Time vs. Matrix Size for Specific Widths')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

