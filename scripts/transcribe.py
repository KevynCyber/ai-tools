import wave
import pyaudio
import os
import argparse
from faster_whisper import WhisperModel

# Constants
BLUE = '\033[94m'
RESET_COLOR = '\033[0m'
MODEL_SIZE = "distil-large-v3"

class AudioTranscriber:
    def __init__(self, model_size=MODEL_SIZE):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.p = pyaudio.PyAudio()
    
    def get_input_device_index(self):
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and 'stereo' in info['name'].lower():
                return i
        return None
    
    def record_audio(self, stream, file_path, seconds=1):
        frames = []
        for _ in range(0, int(16000/1024 * seconds)):
            frames.append(stream.read(1024))
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
    
    # TODO: Add speaker diarization
    def transcribe_audio(self, audio_path):
        segments, _ = self.model.transcribe(audio_path)
        return ''.join(segment.text for segment in segments)
    
    def live_transcribe(self):
        print("Getting input device index...")
        stream = self.p.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=16000,
                           input=True,
                           input_device_index=self.get_input_device_index(),
                           frames_per_buffer=1024)
        
        transcript = ""
        
        try:
            print("Stream opened successfully.")
            while True:
                chunk_file = "temp_chunk.wav"
                self.record_audio(stream, chunk_file)
                text = self.transcribe_audio(chunk_file)
                print(BLUE + text + RESET_COLOR)
                os.remove(chunk_file)
                transcript += text
        except KeyboardInterrupt:
            print("Stopping...")
            with open("transcription.txt", "w") as f:
                f.write(transcript)
            print("LOG:" + transcript)
        finally:
            stream.stop_stream()
            stream.close()
    
    def transcribe_file(self, input_file):
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return
        
        text = self.transcribe_audio(input_file)
        print(BLUE + text + RESET_COLOR)
        
        output_file = os.path.splitext(input_file)[0] + "_transcription.txt"
        with open(output_file, "w") as f:
            f.write(text)
        print(f"Transcription saved to {output_file}")

    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Run live transcription')
    parser.add_argument('--input', type=str, help='Input WAV file to transcribe')
    args = parser.parse_args()
    
    transcriber = AudioTranscriber()
    
    if args.live:
        print('Transcribing live...')
        transcriber.live_transcribe()
    elif args.input:
        print('Transcribing file...')
        transcriber.transcribe_file(args.input)
    else:
        print("Please specify either --live for live transcription or --input for file transcription")

if __name__ == "__main__":
    main()