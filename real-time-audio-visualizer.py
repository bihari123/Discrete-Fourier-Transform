import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq

class AudioVisualizer:
    def __init__(self):
        # Audio parameters
        self.CHUNK = 1024 * 2  # Number of audio samples per frame
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100  # Sampling rate (Hz)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK
        )
        
        # Initialize the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Real-Time Audio Visualizer')
        
        # Time domain plot
        self.line_time, = self.ax1.plot([], [], lw=2)
        self.ax1.set_title('Time Domain')
        self.ax1.set_xlabel('Sample')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_xlim(0, self.CHUNK)
        self.ax1.grid(True)
        
        # Frequency domain plot
        self.line_freq, = self.ax2.plot([], [], lw=2)
        self.ax2.set_title('Frequency Domain')
        self.ax2.set_xlabel('Frequency (Hz)')
        self.ax2.set_ylabel('Magnitude')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_xlim(0, self.RATE // 2)
        self.ax2.grid(True)
        
        # Compute frequency array once
        self.freq_array = fftfreq(self.CHUNK, 1 / self.RATE)
        self.freq_array = self.freq_array[:self.CHUNK // 2]
        
        plt.tight_layout()

    def get_audio_data(self):
        """Read audio data from stream and convert to numpy array"""
        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.float32)

    def animate(self, frame):
        """Animation function called for each frame"""
        # Get audio data
        audio_data = self.get_audio_data()
        
        # Update time domain plot
        self.line_time.set_data(np.arange(len(audio_data)), audio_data)
        
        # Compute FFT and update frequency domain plot
        yf = fft(audio_data)
        yf = 2.0/self.CHUNK * np.abs(yf[:self.CHUNK//2])
        yf = yf / np.max(yf) if np.max(yf) > 0 else yf  # Normalize
        self.line_freq.set_data(self.freq_array, yf)
        
        return self.line_time, self.line_freq

    def run(self):
        """Start the visualization"""
        try:
            ani = FuncAnimation(
                self.fig, 
                self.animate, 
                interval=30,  # Update every 30ms
                blit=True
            )
            plt.show()
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        plt.close()

if __name__ == "__main__":
    visualizer = AudioVisualizer()
    visualizer.run()
