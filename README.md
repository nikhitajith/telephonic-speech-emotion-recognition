Speech Emotion Recognition on Telephonic Audio
A real-time pipeline that detects the speaker's emotion in telephone-quality speech and transcribes what was said. Speech is converted to telephone bandwidth, Wav2Vec2 embeddings are classified into five emotions, and Whisper speech-to-text plus keyword extraction captures the content of the call.

📄 Published paper: [Real-Time Speech Emotion Recognition with
Keyword Extraction Under Telephonic Channel
Conditions], [ICISST 2026]. Authors: Nikhita Ajith, Jyothish Lal G, Amrita Vishwa Vidyapeetham, Coimbatore

Overview
Call centres, helplines and voice assistants mostly receive narrowband telephone audio rather than clean studio recordings. This project tests whether emotion can be recognised reliably under those conditions. It combines:

Telephonic simulation: resampling to 8 kHz and a 4th-order Butterworth band-pass filter (300–3400 Hz)
Emotion recognition: mean-pooled facebook/wav2vec2-base embeddings classified with a Random Forest and a feed-forward neural network
Speech-to-text: OpenAI Whisper (base)
Keyword extraction: NLTK tokenisation and stop-word removal
Real-time mode: records 5 s from the microphone and outputs the emotion and keywords
Pipeline
Microphone / .wav
      │
      ▼
Telephonic conversion (8 kHz, 300–3400 Hz band-pass)
      │
      ├──► Wav2Vec2 embeddings ──► Classifier ──► Emotion
      │
      └──► Whisper ASR ──► Transcript ──► Keyword extraction ──► Keywords
Example output:

--- FINAL OUTPUT ---
Emotion: neutral
Keywords: ['today', 'news']
Datasets
Dataset	Language	Samples used	Emotions
Indian Emotional Speech Corpora (IESC)	English	600 (120 per class)	anger, fear, happy, neutral, sad
Hindi emotional speech dataset	Hindi	filtered to the same 5 emotions	used for cross-lingual testing
The datasets are not included in this repo. Download them and place them as shown below:

data/
├── IESC/      # <speaker>/<emotion>/*.wav
└── hindi/     # <speaker>/<session>/<emotion>/*.wav
Results
IESC (English), 80/20 train/test split, 5 classes

Model	Accuracy
Wav2Vec2 + Random Forest	80.0%
Wav2Vec2 + Neural Network (256→128→5)	81.7%
Per-class F1 (Random Forest): anger 0.83 · fear 0.83 · happy 0.76 · neutral 0.80 · sad 0.79

Cross-lingual test (trained on English IESC, tested on Hindi telephonic audio): 29.9%. Emotion cues learned from one language transfer poorly to another, so language-specific or multilingual training is needed.

Repository structure
├── Speech_Emotion_Recognition_Telephonic.ipynb   # full pipeline notebook
├── data/                                         # datasets (not tracked)
├── requirements.txt
└── README.md
How to run
bash
git clone https://github.com/<your-username>/telephonic-speech-emotion-recognition.git
cd telephonic-speech-emotion-recognition
pip install -r requirements.txt
jupyter notebook Speech_Emotion_Recognition_Telephonic.ipynb
requirements.txt:

torch
transformers
openai-whisper
librosa
soundfile
scipy
scikit-learn
nltk
sounddevice
matplotlib
numpy
Real-time recording needs a working microphone. Whisper also needs FFmpeg installed.

Visualisations
The notebook includes a waveform, original vs. telephonic spectrograms, MFCCs, a pitch contour (F0) and signal energy for sample utterances.

Future work
Fine-tune Wav2Vec2 end-to-end on telephonic speech
Multilingual training to improve performance on Hindi
Fuse acoustic emotion with sentiment from the transcript
Citation
If you use this work, please cite:

bibtex
@inproceedings{ajith_ser_telephonic,
  title     = {Real-Time Speech Emotion Recognition with
Keyword Extraction Under Telephonic Channel
Conditions},
  author    = {Nikhita Ajith and Jyothish Lal G},
  booktitle = {International Conference on Intelligent Systems and Sustainable Technologies},
  year      = {[2026]}
}
