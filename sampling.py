import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

"""
sample the audio signal from computer mic and represent the audio signals in discrete numerical values
Here, for the sake of simplicity, I am recording audio in monochannel (1-Dimension)
"""

def record_audio(duration=5, sample_rate=44100):
    """
    Record mono audio from microphone
    
    Parameters:
    duration (int): Recording duration in seconds
    sample_rate (int): Sample rate in Hz
    
    Returns:
    tuple: (recording data, sample rate, time points)
    """
    
    print("Recording started...")
    # Note the channels=1 parameter for mono recording
    recording = sd.rec(
        frames=int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,  # Changed to 1 channel
        dtype=np.int16
    )
    
    sd.wait()
    print("Recording finished!")
    
    # Flatten the array to make it 1D
    """
    Even when we record with channels=1 (mono), sounddevice.rec() still returns a 2D array with shape (number_of_samples, 1). It's like having a table with many rows but just one column. For example:
    # Without flatten(), the recording array looks like this:
    recording = [
    [120],    # First sample
    [145],    # Second sample
    [130],    # Third sample
    # ... and so on
    ]

    # After flatten(), it becomes:
    recording = [120, 145, 130, ...]  # Simple 1D array

    In here there might be some negative values also, The audio  waves comprise of two components: Compression and rarefraction. The positive values is representing the waves pushing out (compression) and the negative values represent waves pulling in.

    +32767    |     ╭─╮
              |    ╭╯ ╰╮
         0    |────╯   ╰────── (baseline)
              |   ╭╯   ╭╮
    -32768    |  ╭╯    ╰╯
    """
   
    recording = recording.flatten()
    
    # Create time points array
    time_points = np.linspace(0, duration, len(recording))
    
    return recording, sample_rate, time_points

def print_non_zero_values(recording, time_points):
    """
    Print only non-zero values from the audio signal
    """
    # Create mask for non-zero values
    non_zero_mask = recording != 0
    
    # Get non-zero values
    non_zero_times = time_points[non_zero_mask]
    non_zero_values = recording[non_zero_mask]
    
    print("\nNon-zero values in the signal:")
    print(f"Total non-zero samples: {np.sum(non_zero_mask)} out of {len(recording)}")
    
    if np.sum(non_zero_mask) > 0:
        # Print first 20 non-zero values
        print("\nFirst 20 non-zero values:")
        print("Time (s) | Amplitude")
        print("-" * 25)
        for i in range(min(20, len(non_zero_times))):
            print(f"{non_zero_times[i]:.3f} | {non_zero_values[i]:6d}")
        
        # Calculate percentage of non-zero values
        percentage = (np.sum(non_zero_mask) / len(recording)) * 100
        print(f"\nPercentage of non-zero values: {percentage:.2f}%")
        
        # Print statistics for non-zero values
        print("\nNon-zero values statistics:")
        print(f"Min non-zero amplitude: {np.min(non_zero_values)}")
        print(f"Max non-zero amplitude: {np.max(non_zero_values)}")
        print(f"Mean of non-zero values: {np.mean(non_zero_values):.2f}")
    else:
        print("No non-zero values found in the recording")

def save_audio(filename, recording, sample_rate):
    """Save recorded audio to WAV file"""
    wav.write(filename, sample_rate, recording)
    print(f"Audio saved to {filename}")

def save_non_zero_values(recording, time_points, filename="non_zero_audio_values.txt"):
    """Save non-zero values to text file"""
    non_zero_mask = recording != 0
    
    # Get non-zero values
    non_zero_times = time_points[non_zero_mask]
    non_zero_values = recording[non_zero_mask]
    
    # Save to file
    with open(filename, 'w') as f:
        f.write("Time(s),Amplitude\n")
        for t, v in zip(non_zero_times, non_zero_values):
            f.write(f"{t:.6f},{v}\n")
    
    print(f"Non-zero values saved to {filename}")

if __name__ == "__main__":
    # Set recording duration (in seconds)
    duration = 5
    
    # Record audio and get discrete values
    recording, sample_rate, time_points = record_audio(duration)
    
    # Save audio file
    save_audio("recorded_audio.wav", recording, sample_rate)
    
    # Print non-zero values
    print_non_zero_values(recording, time_points)
    
    # Save non-zero values to file
    save_non_zero_values(recording, time_points)
