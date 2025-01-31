"""
Noise reduction implementation with detailed educational comments.
This version uses sounddevice for reliable audio recording and playback.
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

class NoiseReduction:
    def __init__(self, sample_rate=44100):
        """
        Initialize noise reduction parameters.
        
        The audio signal will be processed in overlapping frames:
        - frame_size: Number of samples in each frame
        - hop_size: Number of samples to advance between frames
        - sample_rate: Number of audio samples per second
        """
        self.sample_rate = sample_rate
        self.frame_size = 2048
        self.hop_size = self.frame_size // 2
        
        print("\nNoise Reduction initialized:")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Frame Size: {self.frame_size} samples")
        print(f"Hop Size: {self.hop_size} samples")

    def record_audio(self, duration, is_noise=False):
        """
        Record audio using sounddevice.
        
        Parameters:
        - duration: Recording length in seconds
        - is_noise: Whether this is a noise profile recording
        
        This function records audio data as an array of float32 values
        in the range [-1.0, 1.0]. Each sample represents the amplitude
        of the audio signal at that point in time.
        """
        purpose = "noise profile" if is_noise else "main audio"
        print(f"\nRecording {purpose} for {duration} seconds...")
        input("Press Enter to start recording...")
        print("Recording...")
        
        recording = sd.rec(
            frames=int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        sd.wait()
        print("Recording finished!")
        
        if recording.ndim > 1:
            recording = recording.flatten()
            
        return recording

    def process_frames(self, signal):
        """
        Split signal into overlapping frames with Hanning window.
        
        The signal is divided into overlapping frames like this:
        Signal:     [0....1024....2048....3072....4096]
        Frame 1:    [0........2048]
        Frame 2:        [1024........3072]
        Frame 3:            [2048........4096]
        
        #########################################################
        #              Hanning Window Explanation               #
        #                                                       #
        #  1. What is Hanning Window?                          #
        #     Shape:    1.0 |    ∩                            #
        #               0.8 |   /  \                          #
        #               0.6 |  /    \                         #
        #               0.4 | /      \                        #
        #               0.2 |/        \                       #
        #               0.0 |__________\                      #
        #                    0   N/2   N                      #
        #                                                       #
        #  2. Why Use It?                                      #
        #     Without Window:                                  #
        #     Frame1: [...1 1 1]|                             #
        #     Frame2:           |[1 1 1...]                   #
        #              ^ Sharp edge creates artifacts          #
        #                                                       #
        #     With Window:                                     #
        #     Frame1: [...1 0.5 0]|                           #
        #     Frame2:             |[0 0.5 1...]               #
        #              ^ Smooth transition                     #
        #                                                       #
        #  3. Overlapping Process:                            #
        #     Frame1:   /\                                    #
        #     Frame2:     /\                                  #
        #     Frame3:       /\                                #
        #     Result:  /\/\/\    (50% overlap)               #
        #########################################################
        """
        num_frames = 1 + (len(signal) - self.frame_size) // self.hop_size
        frames = np.zeros((num_frames, self.frame_size))
        window = np.hanning(self.frame_size)
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frames[i] = signal[start:end] * window
            
        return frames

    def get_noise_profile(self, noise_signal):
        """
        Estimate noise frequency profile.
        
        Background noise tends to be statistically constant.
        By averaging multiple frames in the frequency domain,
        we get a stable estimate of the noise spectrum.
        Working in frequency domain lets us target specific frequencies.
        """
        if len(noise_signal) < self.frame_size:
            raise ValueError("Noise signal must be at least as long as frame_size")
            
        frames = self.process_frames(noise_signal)
        spectra = np.fft.rfft(frames, axis=1)  # Convert to frequency domain
        magnitude_spectrum = np.mean(np.abs(spectra), axis=0)  # Average across frames
        return magnitude_spectrum

    def reduce_noise(self, signal, noise_profile, reduction_strength=2.0):
        """
        Apply noise reduction using spectral subtraction.
        
        The process works by:
        1. Splitting signal into overlapping frames
        2. Converting each frame to frequency domain
        3. Subtracting scaled noise profile
        4. Applying temporal smoothing to avoid musical noise
        5. Converting back to time domain
        6. Reconstructing signal with proper normalization
        """
        frames = self.process_frames(signal)
        num_frames = len(frames)
        output = np.zeros(len(signal))
        window = np.hanning(self.frame_size)
        
        prev_magnitude = None
        for i in range(num_frames):
            if i % 10 == 0:
                print(f"Progress: {i/num_frames*100:.1f}%")
                
            # Convert to frequency domain
            spectrum = np.fft.rfft(frames[i])
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Subtract noise profile
            magnitude_clean = magnitude - (noise_profile * reduction_strength)
            magnitude_clean = np.maximum(magnitude_clean, 0)
            
            """
            Apply temporal smoothing to reduce musical noise:
            
            Example with one frequency component:
            Frame1: 0.5    # Current frequency magnitude
            Frame2: 0.8    # Current frequency magnitude
            
            Without smoothing:  [0.5, 0.8]           # Abrupt change
            With smoothing:     [0.5, 0.74]          # Smoother change
                               # 0.74 = 0.8*0.8 + 0.2*0.5
                               
            This prevents rapid on/off switching of frequencies that would
            sound like: *beep* *silence* *beep* *silence*
            """
            if prev_magnitude is not None:
                magnitude_clean = 0.8 * magnitude_clean + 0.2 * prev_magnitude
            prev_magnitude = magnitude_clean.copy()
            
            # Convert back to time domain
            spectrum_clean = magnitude_clean * np.exp(1j * phase)
            frame_clean = np.real(np.fft.irfft(spectrum_clean))
            
            # Add to output with overlap
            start = i * self.hop_size
            end = start + self.frame_size
            output[start:end] += frame_clean * window
        
        # Normalize for overlap-add
        """
        Each frame is multiplied by window twice:
        - Once during framing
        - Once during reconstruction
        
        Without normalization:
        signal = original * window * window  # Too much attenuation
        
        With normalization:
        signal = (original * window * window) / window_sum  # Proper amplitude
        """
        norm_buffer = np.zeros_like(output)
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            norm_buffer[start:end] += window
            
        norm_buffer = np.maximum(norm_buffer, 1e-10)  # Avoid divide by zero
        output /= norm_buffer
        
        print("Noise reduction complete!")
        return output

    def save_audio(self, audio_data, filename):
        """Save audio to WAV file using 16-bit PCM format."""
        audio_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
        wav.write(filename, self.sample_rate, audio_int16)
        print(f"\nSaved to {filename}")

    def play_audio(self, audio_data, description=""):
        """Play audio using sounddevice."""
        if description:
            print(f"\nPlaying {description}...")
        input("Press Enter to start playback...")
        print("Playing...")
        sd.play(audio_data, self.sample_rate)
        sd.wait()
        print("Playback finished!")

def main():
    # Initialize
    nr = NoiseReduction(sample_rate=44100)
    
    try:
        # Record noise profile (2 seconds)
        print("\nStep 1: Record background noise")
        print("Please be quiet during this recording!")
        noise = nr.record_audio(duration=2.0, is_noise=True)
        nr.save_audio(noise, "noise.wav")
        
        # Record main audio (5 seconds)
        print("\nStep 2: Record main audio")
        print("Speak normally during this recording")
        audio = nr.record_audio(duration=5.0)
        nr.save_audio(audio, "original.wav")
        
        # Process
        print("\nStep 3: Process audio")
        noise_profile = nr.get_noise_profile(noise)
        cleaned = nr.reduce_noise(audio, noise_profile)
        nr.save_audio(cleaned, "cleaned.wav")
        
        # Playback
        print("\nStep 4: Compare results")
        nr.play_audio(audio, "original audio")
        time.sleep(0.5)
        nr.play_audio(cleaned, "cleaned audio")
        
        print("\nFiles saved:")
        print("- noise.wav (noise profile)")
        print("- original.wav (original recording)")
        print("- cleaned.wav (noise reduced)")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if your microphone is connected")
        print("2. Check if it's set as the default recording device")
        print("3. Try adjusting your system's audio settings")

if __name__ == "__main__":
    main()
