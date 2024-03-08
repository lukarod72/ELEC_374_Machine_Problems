import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV files
block_data = pd.read_csv('BLOCK_width_timing_data.csv')
tile_data = pd.read_csv('tile_width_timing_data.csv')

# Filter the data for width 25
block_data_25 = block_data[block_data['BlockWidth'] == 25]
tile_data_25 = tile_data[tile_data['TileWidth'] == 25]

# Prepare the plot
plt.figure(figsize=(10, 6))

# Plot data for BlockWidth 25
plt.errorbar(block_data_25['MatrixSize'], block_data_25['GPUAverage'], yerr=block_data_25['GPUError'], fmt='-o', label='Block Width 25')

# Plot data for TileWidth 25
plt.errorbar(tile_data_25['MatrixSize'], tile_data_25['GPUAverage'], yerr=tile_data_25['GPUError'], fmt='--^', label='Tile Width 25')

# Configure the plot
plt.title('GPU Average Time vs. Matrix Size for Width 25')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
