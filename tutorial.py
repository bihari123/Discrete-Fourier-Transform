# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats, signal

# 1. NumPy Basics
# Creating arrays
array_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", array_1d)

array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", array_2d)

zeros = np.zeros((3, 3))
print("\nZeros Array:\n", zeros)

ones = np.ones((2, 4))
print("\nOnes Array:\n", ones)

random_array = np.random.rand(3, 3)
print("\nRandom Array:\n", random_array)

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("\nArray 1:", arr1)
print("Array 2:", arr2)

addition = arr1 + arr2
print("Addition:", addition)

multiplication = arr1 * arr2
print("Element-wise multiplication:", multiplication)

dot_product = np.dot(arr1, arr2)
print("Dot product:", dot_product)

# Array manipulation
reshaped = array_1d.reshape(5, 1)
print("\nReshaped Array:\n", reshaped)

transposed = array_2d.T
print("\nTransposed Array:\n", transposed)

concatenated = np.concatenate((arr1, arr2))
print("\nConcatenated Array:", concatenated)

# 2. Matplotlib Basics
# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('Simple Line Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
plt.figure(figsize=(10, 6))
plt.scatter(x_scatter, y_scatter, c='red', alpha=0.5)
plt.title('Scatter Plot')
plt.grid(True)
plt.show()

# Subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, np.sin(x))
ax1.set_title('Sin(x)')
ax1.grid(True)
ax2.plot(x, np.cos(x))
ax2.set_title('Cos(x)')
ax2.grid(True)
plt.tight_layout()
plt.show()

# 3. SciPy Basics
# Optimization
def objective(x):
    return x**2 + 10*np.sin(x)

result = optimize.minimize(objective, x0=0)
print("\nOptimization Result:")
print("Minimum found at x =", result.x[0])
print("Minimum value =", result.fun)

# Signal processing
t = np.linspace(0, 1, 1000)
signal_data = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
fourier = np.fft.fft(signal_data)
freq = np.fft.fftfreq(len(t), t[1] - t[0])

plt.figure(figsize=(10, 6))
plt.plot(freq[:len(freq)//2], np.abs(fourier)[:len(freq)//2])
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Statistics
normal_dist = stats.norm.rvs(loc=0, scale=1, size=1000)
print("\nNormal Distribution Statistics:")
print("Mean:", np.mean(normal_dist))
print("Standard Deviation:", np.std(normal_dist))

kde = stats.gaussian_kde(normal_dist)
x_range = np.linspace(min(normal_dist), max(normal_dist), 100)
plt.figure(figsize=(10, 6))
plt.plot(x_range, kde(x_range))
plt.title('KDE of Normal Distribution')
plt.grid(True)
plt.show()

# Advanced Example: Signal Processing and Visualization
# Generate a noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2*np.pi*10*t)
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = clean_signal + noise

# Apply a lowpass filter
b, a = signal.butter(4, 0.1, 'low')
filtered_signal = signal.filtfilt(b, a, noisy_signal)

print("\nSignal Statistics:")
print("Clean Signal Mean:", np.mean(clean_signal))
print("Noisy Signal Mean:", np.mean(noisy_signal))
print("Filtered Signal Mean:", np.mean(filtered_signal))

# Visualize results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, noisy_signal, 'b-', label='Noisy')
ax1.plot(t, clean_signal, 'g-', label='Clean')
ax1.set_title('Original Signals')
ax1.legend()
ax1.grid(True)

ax2.plot(t, filtered_signal, 'r-', label='Filtered')
ax2.plot(t, clean_signal, 'g-', label='Clean')
ax2.set_title('Filtered vs Clean Signal')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
