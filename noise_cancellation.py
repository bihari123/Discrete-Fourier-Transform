import numpy as np
import pyaudio
import wave
import time
from scipy.io import wavfile
import sounddevice as sd

class NoiseReduction:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.frame_size = 2048
        self.hop_size = self.frame_size // 2

    def hanning_window(self, size: int) -> np.ndarray:
        return np.hanning(size)

    def frame_signal(self, signal: np.ndarray) -> np.ndarray:
        num_frames = 1 + (len(signal) - self.frame_size) // self.hop_size
        frames = np.zeros((num_frames, self.frame_size))
        window = self.hanning_window(self.frame_size)

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frames[i] = signal[start:end] * window

        return frames

    def estimate_noise_profile(self, noise_signal: np.ndarray) -> np.ndarray:
        if len(noise_signal) < self.frame_size:
            raise ValueError("Noise signal must be at least as long as frame_size")

        frames = self.frame_signal(noise_signal)
        spectra = np.fft.rfft(frames, axis=1)
        magnitude_spectrum = np.mean(np.abs(spectra), axis=0)
        
        return magnitude_spectrum

    def reduce_noise(self, signal: np.ndarray, noise_profile: np.ndarray, 
                    reduction_factor: float = 2.0) -> np.ndarray:
        frames = self.frame_signal(signal)
        num_frames = len(frames)
        output = np.zeros(len(signal))
        
        prev_magnitude = None
        for i in range(num_frames):
            spectrum = np.fft.rfft(frames[i])
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            magnitude_clean = magnitude - (noise_profile * reduction_factor)
            magnitude_clean = np.maximum(magnitude_clean, 0)
            
            if prev_magnitude is not None:
                magnitude_clean = 0.8 * magnitude_clean + 0.2 * prev_magnitude
            prev_magnitude = magnitude_clean.copy()
            
            spectrum_clean = magnitude_clean * np.exp(1j * phase)
            frame_clean = np.real(np.fft.irfft(spectrum_clean))
            
            start = i * self.hop_size
            end = start + self.frame_size
            output[start:end] += frame_clean * self.hanning_window(self.frame_size)

        norm_buffer = np.zeros_like(output)
        window = self.hanning_window(self.frame_size)
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            norm_buffer[start:end] += window
            
        norm_buffer = np.maximum(norm_buffer, 1e-10)
        output /= norm_buffer

        return output

def record_audio(filename, duration, sample_rate=44100, delay=3):
    chunk = 1024
    format = pyaudio.paFloat32
    channels = 1

    p = pyaudio.PyAudio()

    print(f"Recording will start in {delay} seconds...")
    time.sleep(delay)
    print("Recording...")

    stream = p.open(format=format,
                   channels=channels,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=chunk)

    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert frames to numpy array
    audio_data = np.concatenate(frames)
    
    # Save as WAV file
    audio_data_int16 = np.clip(audio_data * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_data_int16)
    
    return audio_data, sample_rate

def play_audio(audio_data, sample_rate):
    print("Playing audio...")
    sd.play(audio_data, sample_rate)
    sd.wait()  # Wait until the audio is finished playing

if __name__ == "__main__":
    # Parameters
    SAMPLE_RATE = 44100
    RECORD_DURATION = 6  # seconds
    DELAY = 3  # seconds
    
    # Initialize noise reduction
    nr = NoiseReduction(sample_rate=SAMPLE_RATE)
    
    # Record audio
    print("We'll record background noise first (1 second)")
    noise_data, _ = record_audio("noise.wav", duration=1, delay=1)
    
    print("\nNow we'll record your main audio")
    audio_data, _ = record_audio("original.wav", duration=RECORD_DURATION, delay=DELAY)
    
    # Process audio
    print("Processing audio...")
    noise_profile = nr.estimate_noise_profile(noise_data)
    clean_audio = nr.reduce_noise(audio_data, noise_profile, reduction_factor=2.0)
    
    # Save cleaned audio
    clean_audio_int16 = np.clip(clean_audio * 32768.0, -32768, 32767).astype(np.int16)
    wavfile.write("clean.wav", SAMPLE_RATE, clean_audio_int16)
    
    # Playback
    print("\nPlaying original audio...")
    play_audio(audio_data, SAMPLE_RATE)
    
    time.sleep(1)  # Wait a second between playbacks
    
    print("\nPlaying cleaned audio...")
    play_audio(clean_audio, SAMPLE_RATE)
