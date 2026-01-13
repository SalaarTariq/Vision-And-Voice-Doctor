import logging 
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
def transcribe_audio(file_path,timeout=20,phrase_time_limit=None):
    """Simplified function to transcribe audio from a file using SpeechRecognition library.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Listening for audio input...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Transcribing audio...")
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_file(BytesIO(wav_data))
            audio_segment.export(file_path,format="mp3",bitrate="128k")
            logging.info(f"Transcription complete. Saved to path {file_path}")
    except Exception as e:
        logging.error(f"Error during transcription: {e}")

transcribe_audio("patient_voice.mp3",timeout=15,phrase_time_limit=10)
