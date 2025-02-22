import whisper
import pyaudio
import numpy as np

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Initialize PyAudio with system audio (Stereo Mix)
audio = pyaudio.PyAudio()

# Find the index of the "Stereo Mix" or system audio device
stereo_mix_index = None
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    if "Stereo Mix" in device_info["name"] or "What U Hear" in device_info["name"]:
        stereo_mix_index = i
        break

if stereo_mix_index is None:
    raise Exception("Stereo Mix device not found. Make sure it’s enabled in the Recording devices.")

# Set up PyAudio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Open stream for system audio
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, input_device_index=stereo_mix_index,
                    frames_per_buffer=CHUNK)

print("Listening to system audio...")

frames = []

try:
    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, np.int16))

        # Accumulate chunks for a certain duration before transcription
        if len(frames) >= RATE // CHUNK * 5:  # 5 seconds buffer
            audio_chunk = np.concatenate(frames)
            frames = []  # Reset buffer

            # Convert audio_chunk to a float32 numpy array
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0  # Normalize to range [-1, 1]

            # Transcription with Whisper
            transcription = whisper_model.transcribe(audio_chunk)
            print("Transcription:", transcription["text"])

except KeyboardInterrupt:
    print("Stopping transcription.")

finally:
    # Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()
