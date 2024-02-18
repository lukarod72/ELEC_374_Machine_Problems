import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
data = pd.read_csv('timing_data_with_errors.csv', skiprows=range(6,12))  # Adjust skiprows to fit your CSV structure if needed

# Plot Home to Device vs Device to Home
plt.figure(figsize=(10, 7))
plt.errorbar(data['TileWidth'], data['HomeToDevice'], yerr=data['HTDError'], label='Home to Device', fmt='-o', capsize=5)
plt.errorbar(data['TileWidth'], data['DeviceToHome'], yerr=data['DTHError'], label='Device to Home', fmt='-^', capsize=5)
plt.title('Home to Device vs Device to Home Timings')
plt.xlabel('Tile Width')
plt.ylabel('Timing (ms)')
plt.legend()
plt.grid(True)
plt.show()

# Plot GPU vs CPU
plt.figure(figsize=(10, 7))
plt.errorbar(data['TileWidth'], data['GPU'], yerr=data['GPUError'], label='GPU Runtime', fmt='-o', capsize=5)
plt.errorbar(data['TileWidth'], data['CPU'], yerr=data['CPUError'], label='CPU Runtime', fmt='-^', capsize=5)
plt.title('GPU vs CPU Runtime')
plt.xlabel('Tile Width')
plt.ylabel('Runtime (ms)')
plt.legend()
plt.grid(True)
plt.show()

# Load the second section for GPU times per tile width change
# Assuming the second section starts at line 7, adjust accordingly
gpu_data = pd.read_csv('timing_data_with_errors.csv', skiprows=1)

# Plot GPU Runtime per Tile Width Change
plt.figure(figsize=(10, 7))
plt.errorbar(gpu_data['TileWidth'], gpu_data['GPUAverage'], yerr=gpu_data['GPUError'], label='GPU Runtime', fmt='-o', capsize=5)
plt.title('GPU Runtime vs Tile Width')
plt.xlabel('Tile Width')
plt.ylabel('GPU Runtime (ms)')
plt.legend()
plt.grid(True)
plt.show()
