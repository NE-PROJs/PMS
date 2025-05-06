from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch

# Load fine-tuned KinyaWhisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("rw-test01.mp3")

# Convert stereo to mono if necessary
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

# Prepare input
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"])

# Decode output
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Print transcription
print("🗣️ Transcription:", transcription)

# Save transcription into a text file
with open("transcription_output.txt", "w", encoding="utf-8") as f:
    f.write(transcription)

print("Transcription saved to 'transcription_output.txt'")