import os
from gtts import gTTS
from dotenv import load_dotenv # Added this import

def text_to_speech_gtts(input_text, output_file):
    tts = gTTS(text=input_text, lang='en',slow=False)
    tts.save(output_file)

input_text = "Hello, how are you?"
output_file = "output.mp3"
text_to_speech_gtts(input_text, output_file)

# Load .env file explicitly
# Assumes .env is in the SAME directory as this script
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(dotenv_path):
    print(f"Warning: .env not found at {dotenv_path}")
load_dotenv(dotenv_path)

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    # Try loading from parent directory as fallback
    parent_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(parent_env)
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

if ELEVENLABS_API_KEY:
    print(f"Loaded API KEY: {ELEVENLABS_API_KEY[:5]}...{ELEVENLABS_API_KEY[-5:]}")
else:
    print("ERROR: ELEVENLABS_API_KEY not found in environment variables.")
import elevenlabs
from elevenlabs.client import ElevenLabs


def text_to_speech_elvnlab(input_text, output_file):
    client=ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="9BWtsMINqrJLrRacOk9x", # Aria's Voice ID
        output_format="mp3_44100_128",
        model_id="eleven_turbo_v2"
    )
    elevenlabs.save(audio,output_file)

text_to_speech_elvnlab(input_text, output_file="Elevenlabs_output.mp3")