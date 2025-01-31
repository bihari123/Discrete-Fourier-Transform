import numpy as np
import pyaudio
import wave
import time
import sys
from scipy.fft import fft, ifft
from scipy.signal import lfilter
from threading import Event
import warnings

# Filter ALSA warnings
warnings.filterwarnings("ignore", category=UserWarning)

class TomCatEffect:
    def __init__(self):
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.FFT_SIZE = 2048
        self.HOP_SIZE = 512
        
        # Specific Tom Cat voice parameters
        self.PITCH_SHIFT_FACTOR = 1.8
        self.FORMANT_BOOST = 2.0
        self.ECHO_DELAY = int(0.08 * self.RATE)
        self.ECHO_DECAY = 0.3
        
        self.MEOW_FORMANTS = {
            'F1': (600, 800),
            'F2': (1800, 2200),
            'F3': (2800, 3200)
        }
        
        self.stop_recording = Event()
        
        # Initialize PyAudio with error suppression
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            print("Error initializing PyAudio. Please check your audio setup.")
            raise e

    def find_pulse_device(self):
        """Find PulseAudio device."""
        try:
            # Look for Pulse devices
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                if ('pulse' in dev['name'].lower() and 
                    dev['maxInputChannels'] > 0):
                    return i
                
            # If no Pulse device found, look for default device
            default = self.p.get_default_input_device_info()
            return default['index']
        except Exception as e:
            print("Error finding audio device. Please check if PulseAudio is installed:")
            print("sudo apt-get install pulseaudio")
            print("pulseaudio --start")
            raise e

    def init_audio_stream(self, is_input=True):
        """Initialize audio stream with proper error handling."""
        try:
            if is_input:
                device_index = self.find_pulse_device()
                stream = self.p.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   input=True,
                                   input_device_index=device_index,
                                   frames_per_buffer=self.CHUNK)
            else:
                stream = self.p.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   output=True,
                                   frames_per_buffer=self.CHUNK)
            return stream
        except Exception as e:
            print("\nError initializing audio stream. Please try these steps:")
            print("1. Restart PulseAudio:")
            print("   pulseaudio -k")
            print("   pulseaudio --start")
            print("2. Check audio devices:")
            print("   pactl list short sources")
            print("   pactl list short sinks")
            print("3. Make sure you're in the audio group:")
            print("   sudo usermod -a -G pulse,pulse-access $USER")
            raise e

    # [Previous audio processing methods remain unchanged]
    def apply_cat_formants(self, spectrum, magnitudes, phases, freq_resolution):
        """Apply characteristic cat vocal formants."""
        for formant_name, (freq_low, freq_high) in self.MEOW_FORMANTS.items():
            bin_low = int(freq_low / freq_resolution)
            bin_high = int(freq_high / freq_resolution)
            
            boost_factor = self.FORMANT_BOOST
            if formant_name == 'F1':
                boost_factor *= 1.5
            
            formant_mask = np.zeros_like(magnitudes)
            formant_mask[bin_low:bin_high] = np.hanning(bin_high - bin_low) * boost_factor
            magnitudes *= (1 + formant_mask)
        
        return magnitudes * np.exp(1j * phases)

    def add_characteristic_meow(self, audio_array):
        duration = len(audio_array) / self.RATE
        t = np.linspace(0, duration, len(audio_array))
        mod_freq = 3 + np.exp(-t*2) * 15
        modulation = 0.15 * np.sin(2 * np.pi * mod_freq * t)
        return audio_array * (1 + modulation)

    def apply_echo(self, audio_array):
        delayed = np.zeros_like(audio_array)
        delayed[self.ECHO_DELAY:] = audio_array[:-self.ECHO_DELAY] * self.ECHO_DECAY
        
        result = audio_array.copy()
        for i in range(2):
            echo_amp = self.ECHO_DECAY ** (i + 1)
            delay_samples = self.ECHO_DELAY * (i + 1)
            if delay_samples < len(audio_array):
                result[delay_samples:] += audio_array[:-delay_samples] * echo_amp
        
        return result

    def apply_fft_effect(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        output_buffer = np.zeros_like(audio_array)
        window = np.hanning(self.FFT_SIZE)
        freq_resolution = self.RATE / self.FFT_SIZE
        
        for frame in range(0, len(audio_array) - self.FFT_SIZE, self.HOP_SIZE):
            current_frame = audio_array[frame:frame + self.FFT_SIZE] * window
            
            spectrum = fft(current_frame)
            magnitudes = np.abs(spectrum)
            phases = np.angle(spectrum)
            
            shifted_spectrum = np.zeros_like(spectrum, dtype=np.complex128)
            
            for i in range(len(spectrum)):
                new_bin = int(i * self.PITCH_SHIFT_FACTOR)
                if new_bin < len(spectrum):
                    if new_bin * 2 < len(spectrum):
                        shifted_spectrum[new_bin * 2] += magnitudes[i] * 0.4 * np.exp(1j * (phases[i] * 2))
                    shifted_spectrum[new_bin] = magnitudes[i] * np.exp(1j * phases[i])
            
            shifted_spectrum = self.apply_cat_formants(shifted_spectrum, 
                                                     np.abs(shifted_spectrum), 
                                                     np.angle(shifted_spectrum),
                                                     freq_resolution)
            
            processed_frame = np.real(ifft(shifted_spectrum))
            output_buffer[frame:frame + self.FFT_SIZE] += processed_frame * window
        
        output_buffer = self.add_characteristic_meow(output_buffer)
        output_buffer = self.apply_echo(output_buffer)
        
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0:
            output_buffer = output_buffer * 0.95 / max_val
        
        return output_buffer.astype(np.float32).tobytes()

    def record_audio(self):
        """Record audio with improved error handling."""
        stream = self.init_audio_stream(is_input=True)
        print("* recording")
        frames = []

        try:
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if self.stop_recording.is_set():
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            raise e
        finally:
            print("* done recording")
            stream.stop_stream()
            stream.close()

        return b''.join(frames)

    def play_audio(self, audio_data):
        """Play audio with improved error handling."""
        stream = self.init_audio_stream(is_input=False)
        print("* playing")
        
        try:
            for i in range(0, len(audio_data), self.CHUNK):
                chunk = audio_data[i:i + self.CHUNK]
                stream.write(chunk)
        except Exception as e:
            print(f"Error during playback: {str(e)}")
            raise e
        finally:
            print("* done playing")
            stream.stop_stream()
            stream.close()

    def save_audio(self, audio_data, filename="output.wav"):
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(audio_data)
            wf.close()
            print(f"* saved to {filename}")
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")
            raise e

    def cleanup(self):
        """Clean up PyAudio."""
        try:
            self.p.terminate()
        except:
            pass

    def process_audio(self):
        """Main processing function with comprehensive error handling."""
        try:
            audio_data = self.record_audio()
            
            print("* applying Tom Cat effect")
            processed_audio = self.apply_fft_effect(audio_data)
            
            self.play_audio(processed_audio)
            self.save_audio(processed_audio)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("\nTroubleshooting guide:")
            print("1. Check PulseAudio installation and status:")
            print("   sudo apt-get install pulseaudio")
            print("   pulseaudio --check")
            print("   pulseaudio --start")
            print("2. Reset PulseAudio if needed:")
            print("   pulseaudio -k")
            print("   pulseaudio --start")
            print("3. Check audio permissions:")
            print("   sudo usermod -a -G pulse,pulse-access $USER")
            print("   sudo usermod -a -G audio $USER")
            print("4. List audio devices:")
            print("   pactl list short sources")
            print("   pactl list short sinks")
        finally:
            self.cleanup()

def check_audio_setup():
    """Check audio setup before starting."""
    try:
        # Check for pulseaudio
        import subprocess
        result = subprocess.run(['pulseaudio', '--check'], 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            print("Starting PulseAudio...")
            subprocess.run(['pulseaudio', '--start'])
            time.sleep(2)  # Give PulseAudio time to start
    except FileNotFoundError:
        print("PulseAudio not found. Installing recommended packages...")
        print("Please run:")
        print("sudo apt-get update")
        print("sudo apt-get install pulseaudio python3-pyaudio python3-numpy python3-scipy")
        sys.exit(1)

def main():
    print("Enhanced Tom Cat Voice Effect (Ubuntu Version)")
    print("This version includes:")
    print("- Improved PulseAudio support")
    print("- Robust error handling")
    print("- Automatic audio setup checks")
    
    try:
        import numpy
        import scipy
    except ImportError as e:
        print(f"\nError: Missing required package - {str(e)}")
        print("Please install required packages:")
        print("sudo apt-get update")
        print("sudo apt-get install python3-pyaudio python3-numpy python3-scipy")
        sys.exit(1)
    
    check_audio_setup()
    effect = TomCatEffect()
    print("\nRecording will start in 3 seconds...")
    time.sleep(3)
    effect.process_audio()

if __name__ == "__main__":
    main()
