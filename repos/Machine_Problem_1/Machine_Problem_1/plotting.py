import matplotlib.pyplot as plt
import pandas as pd

# Load the matrix size timing data
data = pd.read_csv('matrix_size_timing_data.csv')

# Plot Home to Device vs Device to Home for different matrix sizes
plt.figure(figsize=(10, 7))
plt.errorbar(data['MatrixSize'], data['HomeToDevice'], yerr=data['HTDError'], label='Home to Device', fmt='-o', capsize=5)
plt.errorbar(data['MatrixSize'], data['DeviceToHome'], yerr=data['DTHError'], label='Device to Home', fmt='-^', capsize=5)
plt.title('Timing vs. Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Timing (ms)')
plt.legend()
plt.grid(True)
plt.show()

# Plot GPU vs CPU for different matrix sizes
plt.figure(figsize=(10, 7))
plt.errorbar(data['MatrixSize'], data['GPU'], yerr=data['GPUError'], label='GPU Runtime', fmt='-o', capsize=5)
plt.errorbar(data['MatrixSize'], data['CPU'], yerr=data['CPUError'], label='CPU Runtime', fmt='-^', capsize=5)
plt.title('GPU vs CPU Runtime vs. Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Runtime (ms)')
plt.legend()
plt.grid(True)
plt.show()

# Load the tile width timing data
tile_data = pd.read_csv('tile_width_timing_data.csv')

# Plot GPU Runtime per Tile Width Change
plt.figure(figsize=(10, 7))
plt.errorbar(tile_data['TileWidth'], tile_data['GPUAverage'], yerr=tile_data['GPUError'], label='GPU Runtime', fmt='-o', capsize=5)
plt.title('GPU Runtime vs Tile Width')
plt.xlabel('Tile Width')
plt.ylabel('GPU Runtime (ms)')
plt.legend()
plt.grid(True)
plt.show()
