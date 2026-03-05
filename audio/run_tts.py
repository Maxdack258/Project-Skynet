import torch
import soundfile as sf
import subprocess # Built-in module
import speech_recognition as sr_module # Import speech recognition
import whisper # Import whisper
import threading # Threading
import queue # Queue for TTS
import re # RegEx
from ctypes import * # Ctypes for ALSA

# Silencing noisy ALSA warnings
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt): pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass
import requests # API requests
import json # JSON parsing
from qwen_tts import Qwen3TTSModel # Load TTS model

print("Loading Whisper model...")
whisper_model = whisper.load_model("base") # Load base whisper

print("Downloading and loading the model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", 
    device_map="cuda:0", 
    dtype=torch.bfloat16,
)

OPENROUTER_API_KEY = ""

def stream_chat_response(text):
    # Call OpenRouter API with streaming
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "openai/gpt-oss-120b", # Requested model id
            "provider": {"order": ["Clarifai"]}, # Use Clarifai provider
            "stream": True, # Enable response streaming
            "messages": [
                {
                    "role": "system",
                    "content": "You are a calm voice assistant in a spoken conversation. Keep your responses short, natural, and conversational. Do not generate large texts, avoid lists or markdown."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        }),
        stream=True
    )
    
    current_sentence = ""
    for line in response.iter_lines():
        if line:
            chunk = line.decode('utf-8')
            if chunk.startswith("data: "):
                chunk = chunk[6:]
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        content_piece = delta['content']
                        current_sentence += content_piece
                        print(content_piece, end='', flush=True) # Print token stream
                        
                        # Check for sentence ends to yield a chunk
                        if any(p in current_sentence for p in ['.', '!', '?', '\n']):
                            match = re.search(r'[.!?\n]', current_sentence)
                            if match:
                                split_idx = match.end()
                                sentence = current_sentence[:split_idx]
                                yield sentence.strip()
                                current_sentence = current_sentence[split_idx:]
            except json.JSONDecodeError:
                pass

    if current_sentence.strip():
        yield current_sentence.strip() # Yield remaining text

recognizer = sr_module.Recognizer() # Initialize recognizer
# Setup audio thresholds: Wait for volume, stop on silence
recognizer.energy_threshold = 300 # Volume to start
recognizer.pause_threshold = 1.0 # Silence to stop
mic = sr_module.Microphone() # Initialize microphone

while True:
    with mic as source:
        print("\nAdjusting to ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1.0) # Adjust noise
        print("Listening... (Waiting for your voice, stops automatically on silence)")
        audio_data = recognizer.listen(source) # Listen to mic
    
    print("Transcribing with Whisper...")
    with open("temp_input.wav", "wb") as f:
        f.write(audio_data.get_wav_data()) # Save temp audio
        
    result = whisper_model.transcribe("temp_input.wav") # Transcribe audio
    user_text = result['text'].strip() # Get text
    
    if not user_text:
        print("I didn't hear anything.")
        continue
        
    print(f"You said: {user_text}")
    
    if user_text.lower() in ['exit', 'quit', 'stop']:
        print("Exiting...")
        break # Exit loop

    text_queue = queue.Queue()
    audio_queue = queue.Queue()
    
    def tts_generator_worker():
        idx = 0
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None: # Sentinel value to stop
                audio_queue.put(None) # Pass sentinel to playback
                text_queue.task_done()
                break
            
            if text_chunk.strip():
                # Generate audio chunk
                wavs, sr = model.generate_custom_voice(
                    text=text_chunk.strip(),
                    language="English",
                    speaker="Aiden", 
                    instruct="Speak in a calm voice, take your time, not energetic",
                )
                output_file = f"output_{idx}.wav"
                sf.write(output_file, wavs[0], sr) # Save audio chunk
                audio_queue.put(output_file) # Queue for playback
                idx += 1
            text_queue.task_done()

    def tts_playback_worker():
        while True:
            audio_file = audio_queue.get()
            if audio_file is None: # Sentinel value to stop
                audio_queue.task_done()
                break
            
            # Play audio chunk
            subprocess.run(["aplay", audio_file], check=False, stderr=subprocess.DEVNULL)
            audio_queue.task_done()

    # Start audio worker threads
    gen_thread = threading.Thread(target=tts_generator_worker)
    play_thread = threading.Thread(target=tts_playback_worker)
    gen_thread.start()
    play_thread.start()

    print(f"GPT-OSS-120b: ", end='')
    for sentence in stream_chat_response(user_text):
        if sentence:
            text_queue.put(sentence) # Queue text
    print() # New line after full response
    
    text_queue.put(None) # Signal thread to finish
    gen_thread.join() # Wait for generation
    play_thread.join() # Wait for all playback to stop before listening again
