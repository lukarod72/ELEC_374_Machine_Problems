import matplotlib.pyplot as plt
import pandas as pd



# Load data from CSV
data = pd.read_csv('tile_width_timing_data.csv')

# Unique tile widths for plotting separate lines
tile_widths = data['TileWidth'].unique()

# Setup the plot
plt.figure(figsize=(10, 6))

# Iterate over each tile width to plot its data
for tile_width in tile_widths:
    subset = data[data['TileWidth'] == tile_width]
    plt.errorbar(subset['MatrixSize'], subset['GPUAverage'], yerr=subset['GPUError'], label=f'Tile Width {tile_width}')

# Adding plot title and labels
plt.title('GPU Average Time vs. Matrix Size for Different Tile Widths')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()



# Load data from CSV
data = pd.read_csv('tile_width_timing_data_25.csv')

# Unique tile widths for plotting separate lines
tile_widths = data['TileWidth'].unique()

# Setup the plot
plt.figure(figsize=(10, 6))

# Iterate over each tile width to plot its data
for tile_width in tile_widths:
    subset = data[data['TileWidth'] == tile_width]
    plt.errorbar(subset['MatrixSize'], subset['GPUAverage'], yerr=subset['GPUError'], label=f'Tile Width {tile_width}')

# Adding plot title and labels
plt.title('GPU Average Time vs. Matrix Size for 25 as the Tile Width')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
