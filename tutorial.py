# This tutorial demonstrates basic data science operations in Python
# We'll use three main libraries:
# - numpy: for numerical operations and array handling
# - matplotlib: for creating visualizations and plots
# - scipy: for scientific computations like optimization and signal processing

# Import the libraries we'll need for this tutorial
import numpy             # Library for numerical operations
import matplotlib.pyplot # Library for creating plots and visualizations
import scipy.optimize    # For mathematical optimization
import scipy.stats      # For statistical operations
import scipy.signal     # For signal processing operations

# 1. Basic Array Operations with numpy
# ===================================

# Create a simple one-dimensional array (like a list of numbers)
array_1d = numpy.array([1, 2, 3, 4, 5])
print("One-dimensional array:", array_1d)

# Create a two-dimensional array (like a table with rows and columns)
array_2d = numpy.array([
    [1, 2, 3],  # First row
    [4, 5, 6]   # Second row
])
print("\nTwo-dimensional array (2 rows, 3 columns):\n", array_2d)

# Create special arrays filled with zeros
# The (3, 3) means we want a 3x3 table of zeros
zeros_array = numpy.zeros((3, 3))
print("\nArray filled with zeros:\n", zeros_array)

# Create special arrays filled with ones
# The (2, 4) means we want a 2x4 table of ones
ones_array = numpy.ones((2, 4))
print("\nArray filled with ones:\n", ones_array)

# Create an array with random numbers between 0 and 1
random_array = numpy.random.rand(3, 3)
print("\nArray with random numbers:\n", random_array)

# Basic Array Mathematics
# ----------------------
# Let's create two simple arrays to demonstrate mathematical operations
first_array = numpy.array([1, 2, 3])
second_array = numpy.array([4, 5, 6])
print("\nFirst array:", first_array)
print("Second array:", second_array)

# Add the arrays element by element
# [1,2,3] + [4,5,6] = [5,7,9]
addition_result = first_array + second_array
print("Adding the arrays:", addition_result)

# Multiply the arrays element by element
# [1,2,3] * [4,5,6] = [4,10,18]
multiplication_result = first_array * second_array
print("Multiplying the arrays element by element:", multiplication_result)

# Calculate the dot product (sum of element-wise multiplication)
# (1*4 + 2*5 + 3*6) = 4 + 10 + 18 = 32
dot_product_result = numpy.dot(first_array, second_array)
print("Dot product of the arrays:", dot_product_result)

# Array Shape Manipulation
# -----------------------

# Reshape a 1D array into a 2D array (5 rows, 1 column)
reshaped_array = array_1d.reshape(5, 1)
print("\nArray reshaped into 5 rows and 1 column:\n", reshaped_array)

# Transpose a 2D array (swap rows and columns)
transposed_array = array_2d.T  # T stands for transpose
print("\nTransposed array (columns become rows):\n", transposed_array)

# Join two arrays together
concatenated_array = numpy.concatenate((first_array, second_array))
print("\nTwo arrays joined together:", concatenated_array)

# 2. Creating Visualizations with matplotlib
# ========================================

# Create some data for plotting
# linspace creates 100 evenly spaced numbers from 0 to 10
x_values = numpy.linspace(0, 10, 100)
# Calculate sine wave values for each x
y_values = numpy.sin(x_values)

# Create a simple line plot
matplotlib.pyplot.figure(figsize=(10, 6))  # Set the plot size
matplotlib.pyplot.plot(x_values, y_values, label='sin(x)')
matplotlib.pyplot.title('Simple Line Plot of a Sine Wave')
matplotlib.pyplot.xlabel('x axis')
matplotlib.pyplot.ylabel('y axis')
matplotlib.pyplot.legend()  # Show the label we created above
matplotlib.pyplot.grid(True)  # Add a grid to the plot
matplotlib.pyplot.show()  # Display the plot

# Create a scatter plot with random points
x_scatter = numpy.random.rand(50)  # 50 random x values
y_scatter = numpy.random.rand(50)  # 50 random y values
matplotlib.pyplot.figure(figsize=(10, 6))
matplotlib.pyplot.scatter(x_scatter, y_scatter, color='red', alpha=0.5)
matplotlib.pyplot.title('Scatter Plot of Random Points')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()

# Create multiple plots side by side
figure, (left_plot, right_plot) = matplotlib.pyplot.subplots(1, 2, figsize=(12, 4))

# Left plot: sine wave
left_plot.plot(x_values, numpy.sin(x_values))
left_plot.set_title('Sine Wave')
left_plot.grid(True)

# Right plot: cosine wave
right_plot.plot(x_values, numpy.cos(x_values))
right_plot.set_title('Cosine Wave')
right_plot.grid(True)

matplotlib.pyplot.tight_layout()  # Adjust spacing between plots
matplotlib.pyplot.show()

# 3. Scientific Computing with scipy
# ================================


# Signal Processing Example
# ------------------------
# Create a time array from 0 to 1 second (1000 points)
time = numpy.linspace(0, 1, 1000)
# Create a signal with two frequency components
signal_data = numpy.sin(2*numpy.pi*10*time) + numpy.sin(2*numpy.pi*20*time)

# Calculate the frequency spectrum using Fast Fourier Transform
fourier_transform = numpy.fft.fft(signal_data)
frequencies = numpy.fft.fftfreq(len(time), time[1] - time[0])

# Plot the frequency spectrum
matplotlib.pyplot.figure(figsize=(10, 6))
matplotlib.pyplot.plot(
    frequencies[:len(frequencies)//2], 
    numpy.abs(fourier_transform)[:len(frequencies)//2]
)
matplotlib.pyplot.title('Frequency Spectrum of the Signal')
matplotlib.pyplot.xlabel('Frequency (Hz)')
matplotlib.pyplot.ylabel('Magnitude')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()

# Statistics Example
# -----------------
# Generate 1000 random numbers following a normal distribution
normal_distribution = scipy.stats.norm.rvs(loc=0, scale=1, size=1000)
print("\nNormal Distribution Statistics:")
print("Mean (average):", numpy.mean(normal_distribution))
print("Standard Deviation:", numpy.std(normal_distribution))

# Calculate and plot the probability density
kernel_density = scipy.stats.gaussian_kde(normal_distribution)
x_range = numpy.linspace(min(normal_distribution), max(normal_distribution), 100)
matplotlib.pyplot.figure(figsize=(10, 6))
matplotlib.pyplot.plot(x_range, kernel_density(x_range))
matplotlib.pyplot.title('Probability Density of Normal Distribution')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()

# Advanced Example: Signal Processing and Visualization
# =================================================

# Understanding subplot layout:
# matplotlib.pyplot.subplots(rows, columns, figsize=(width, height))
# - rows: number of plots vertically (2 in our case)
# - columns: number of plots horizontally (1 in our case)
# - figsize: (width in inches, height in inches)
#
# Example layout of subplots(2, 1):
# +----------------+  ← figure
# |   top_plot     |  ← subplot(0)
# |                |
# +----------------+
# | bottom_plot    |  ← subplot(1)
# |                |
# +----------------+

# First, create our time values from 0 to 1 second
time = numpy.linspace(0, 1, 1000)

# Understanding sine wave parameters:
# numpy.sin(2 * numpy.pi * frequency * time)
# - 2*numpy.pi: converts from cycles to radians (2π radians = 360 degrees = 1 cycle)
# - frequency: number of complete cycles per second (10 Hz in our case)
# - time: our time array from 0 to 1 second
#
# Example visualization of parameters:
# Amplitude = 1
# Frequency = 10 Hz
# Total cycles in 1 second = 10
#  1┤     /\    /\    /\    /\    /\    /\    /\    /\    /\    /\
#   │    /  \  /  \  /  \  /  \  /  \  /  \  /  \  /  \  /  \  /  \
#  0┼───/────\/────\/────\/────\/────\/────\/────\/────\/────\/────\
#   │  /      \    \    \    \    \    \    \    \    \    \    \
# -1┤ /        \    \    \    \    \    \    \    \    \    \    \
#   └─────────────────────────────────────────────────────────────────→
#   0                          Time (seconds)                         1

clean_signal = numpy.sin(2*numpy.pi*10*time)  # Creates a 10 Hz sine wave
noise = numpy.random.normal(0, 0.5, len(time))  # Random noise with standard deviation 0.5
noisy_signal = clean_signal + noise  # Add noise to the signal

# Design and apply a lowpass filter to remove noise
filter_coefficients = scipy.signal.butter(4, 0.1, 'low')  # Create filter
filtered_signal = scipy.signal.filtfilt(
    filter_coefficients[0], 
    filter_coefficients[1], 
    noisy_signal
)

# Calculate statistics of all signals
print("\nSignal Statistics:")
print("Clean Signal Average:", numpy.mean(clean_signal))
print("Noisy Signal Average:", numpy.mean(noisy_signal))
print("Filtered Signal Average:", numpy.mean(filtered_signal))

# Create a figure with two subplots stacked vertically
# The subplots() function returns two things:
# 1. figure: the overall figure object that contains all subplots
# 2. (top_plot, bottom_plot): tuple of individual subplot objects
#    - top_plot: the upper subplot (index 0)
#    - bottom_plot: the lower subplot (index 1)
figure, (top_plot, bottom_plot) = matplotlib.pyplot.subplots(
    nrows=2,          # 2 rows (stacked vertically)
    ncols=1,          # 1 column
    figsize=(10, 8)   # Width = 10 inches, Height = 8 inches
)

# Top plot: Compare original clean signal with noisy signal
# Plot each line separately so we can give them different colors and labels
top_plot.plot(
    time,           # X-axis values (time points)
    noisy_signal,   # Y-axis values (signal amplitude)
    'blue',         # Line color
    label='Noisy Signal'  # Label for legend
)
top_plot.plot(
    time,
    clean_signal,
    'green',
    label='Clean Signal'
)
top_plot.set_title('Original Signals Comparison')
top_plot.legend()    # Show the color-coded legend
top_plot.grid(True)  # Add grid lines for better readability

# Bottom plot: Compare filtered signal with original clean signal
# This shows how well our filtering worked
bottom_plot.plot(
    time,
    filtered_signal,
    'red',
    label='Filtered Signal'
)
bottom_plot.plot(
    time,
    clean_signal,
    'green',
    label='Clean Signal'
)
bottom_plot.set_title('Filtered vs Clean Signal')
bottom_plot.legend()
bottom_plot.grid(True)

# Adjust spacing between subplots to prevent overlap
matplotlib.pyplot.tight_layout()

# Display the complete figure with both subplots
matplotlib.pyplot.show()
