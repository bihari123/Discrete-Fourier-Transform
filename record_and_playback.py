"""
Simple test script for noise reduction basics.
First, let's verify we can record and play audio correctly.
"""

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

def test_recording(duration=3, sample_rate=44100):
    """Basic test to verify recording works."""
    print(f"\nTesting audio recording for {duration} seconds...")
    
    try:
        # List audio devices
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
        # Get default devices
        print("\nDefault devices:")
        print(f"Input: {sd.query_devices(kind='input')}")
        print(f"Output: {sd.query_devices(kind='output')}")
        
        input("\nPress Enter to start recording...")
        print("Recording...")
        
        # Record audio
        recording = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        sd.wait()
        print("Recording finished!")
        
        # Flatten array if needed
        if recording.ndim > 1: #array dimensions is grater than 0
            recording = recording.flatten()
        
        # Save file
        audio_int16 = np.clip(recording * 32768, -32768, 32767).astype(np.int16)
        wav.write("test_recording.wav", sample_rate, audio_int16)
        print("\nSaved recording to test_recording.wav")
        
        # Play back
        input("\nPress Enter to play back the recording...")
        print("Playing...")
        sd.play(recording, sample_rate)
        sd.wait()
        print("Playback finished!")
        
        return True
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your microphone is connected and working")
        print("2. Check if it's set as the default recording device")
        print("3. Try adjusting your system's audio settings")
        print("4. Make sure you have the required Python packages:")
        print("   pip install sounddevice numpy scipy")
        return False

if __name__ == "__main__":
    print("Audio Recording Test")
    print("===================")
    
    success = test_recording()
    
    if success:
        print("\nBasic audio test completed successfully!")
    else:
        print("\nTest failed. Please check the error messages above.")
