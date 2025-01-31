import numpy as np
import sounddevice as sd
import wave
import time
import sys
from scipy.fft import fft, ifft
from scipy.signal import lfilter
from threading import Event

class TomCatEffect:
    def __init__(self):
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.FFT_SIZE = 2048
        self.HOP_SIZE = 512
        self.BLOCK_SIZE = 2048  # Block size for streaming
        
        # Specific Tom Cat voice parameters
        self.PITCH_SHIFT_FACTOR = 1.8  # Higher pitch for cat-like voice
        
                
        self.stop_recording = Event()


    def apply_fft_effect(self, audio_array):
        """Apply enhanced Tom Cat FFT effect to the audio data."""
        output_buffer = np.zeros_like(audio_array)
        window = np.hanning(self.FFT_SIZE)
        
        # Process frame by frame
        for frame in range(0, len(audio_array) - self.FFT_SIZE, self.HOP_SIZE):
            current_frame = audio_array[frame:frame + self.FFT_SIZE] * window
            
            # FFT processing
            spectrum = fft(current_frame)
            magnitudes = np.abs(spectrum)
            phases = np.angle(spectrum)
            
            # Initialize shifted spectrum
            shifted_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
            
            # Apply pitch shifting and formant enhancement
            for i in range(len(spectrum)):
                new_bin = int(i * self.PITCH_SHIFT_FACTOR)
                if new_bin < len(spectrum):
                    shifted_spectrum[new_bin] = magnitudes[i] * np.exp(1j * phases[i])
                    # shifted_spectrum[new_bin] =magnitudes[i]
            
            
            # Inverse FFT
            processed_frame = np.real(ifft(shifted_spectrum))
            output_buffer[frame:frame + self.FFT_SIZE] += processed_frame * window
        
        
       
        # Normalize
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0:
            output_buffer = output_buffer * 0.95 / max_val
        
        return output_buffer

    def record_audio(self):
        """Record audio using sounddevice."""
        print("* recording")
        audio_data = sd.rec(
            int(self.RATE * self.RECORD_SECONDS),
            samplerate=self.RATE,
            channels=self.CHANNELS,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        print("* done recording")
        return audio_data.flatten()

    def play_audio(self, audio_data):
        """Play processed audio using sounddevice."""
        print("* playing")
        sd.play(audio_data, self.RATE)
        sd.wait()  # Wait until playback is finished
        print("* done playing")

    def save_audio(self, audio_data, filename="output.wav"):
        """Save audio to WAV file."""
        # Convert float32 array to int16 for WAV file
        audio_data_int = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data_int.tobytes())
        
        print(f"* saved to {filename}")

    def process_audio(self):
        """Main processing function."""
        try:
            # Record
            audio_data = self.record_audio()
            
            # Apply effect
            print("* applying Tom Cat effect")
            processed_audio = self.apply_fft_effect(audio_data)
            
            # Play processed audio
            self.play_audio(processed_audio)
            
            # Save to file
            self.save_audio(processed_audio)
            
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    print("Enhanced Tom Cat Voice Effect")
    print("This version includes:")
    print("- Improved pitch shifting")
    print("- Realistic cat formants")
    print("- Characteristic echo effect")
    print("- Meow modulation")
    
    effect = TomCatEffect()
    print("\nRecording will start in 3 seconds...")
    time.sleep(3)
    effect.process_audio()

if __name__ == "__main__":
    main()
