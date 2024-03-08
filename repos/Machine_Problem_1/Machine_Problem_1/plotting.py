import matplotlib.pyplot as plt
import pandas as pd

# Load the matrix size timing data
data = pd.read_csv('timing_data_with_errors.csv')

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


# Load data from CSV
data = pd.read_csv('BLOCK_width_timing_data.csv')

# Unique BLOCK widths for plotting separate lines
BLOCK_widths = data['BlockWidth'].unique()

# Setup the plot
plt.figure(figsize=(10, 6))

# Iterate over each BLOCK width to plot its data
for BLOCK_width in BLOCK_widths:
    subset = data[data['BlockWidth'] == BLOCK_width]
    plt.errorbar(subset['MatrixSize'], subset['GPUAverage'], yerr=subset['GPUError'], label=f'Block Width {BLOCK_width}')

# Adding plot title and labels
plt.title('GPU Average Time vs. Matrix Size for Different Block Widths')
plt.xlabel('Matrix Size')
plt.ylabel('GPU Average Time (ms)')
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
