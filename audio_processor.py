# from moviepy.editor import VideoFileClip
# import json
# from faster_whisper import WhisperModel

# class AudioProcessor:
#     def __init__(self, config):
#         self.config = config
#         self.model = WhisperModel(config.MODEL_SIZE)
    
#     def extract_audio(self, video_path):
#         """Extract audio from video file"""
#         video_clip = VideoFileClip(video_path)
#         audio_clip = video_clip.audio
#         audio_path = "temp_audio.mp3"
#         audio_clip.write_audiofile(audio_path)
#         audio_clip.close()
#         return audio_path

#     def transcribe_audio(self, audio_path):
#         """Transcribe audio to text with timestamps"""
#         segments, _ = self.model.transcribe(audio_path, word_timestamps=True)
#         word_level_info = []
        
#         for segment in segments:
#             for word in segment.words:
#                 word_level_info.append({
#                     'word': word.word,
#                     'start': word.start,
#                     'end': word.end
#                 })
        
#         return word_level_info

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel

class AudioExtractor(ABC):
    """Abstract base class for audio extraction"""
    @abstractmethod
    def extract(self, video_path: str) -> str:
        """Extract audio from video and return audio path"""
        pass

class MoviePyAudioExtractor(AudioExtractor):
    """MoviePy implementation of audio extraction"""
    def extract(self, video_path: str) -> str:
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_path = "temp_audio.mp3"
            audio_clip.write_audiofile(audio_path)
            audio_clip.close()
            video_clip.close()
            return audio_path
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio: {str(e)}")

class Transcriber(ABC):
    """Abstract base class for transcription"""
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio to text with timestamps"""
        pass

class WhisperTranscriber(Transcriber):
    """Whisper implementation of transcription"""
    def __init__(self, model_size: str):
        self.model = WhisperModel(model_size)

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            segments, _ = self.model.transcribe(audio_path, word_timestamps=True)
            return [
                {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end
                }
                for segment in segments
                for word in segment.words
            ]
        except Exception as e:
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")

class AudioProcessor:
    """Main audio processing coordinator"""
    def __init__(self, extractor: AudioExtractor, transcriber: Transcriber):
        self.extractor = extractor
        self.transcriber = transcriber

    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process video to get word-level transcription"""
        audio_path = self.extractor.extract(video_path)
        return self.transcriber.transcribe(audio_path)

# Custom exceptions
class AudioExtractionError(Exception):
    """Raised when audio extraction fails"""
    pass

class TranscriptionError(Exception):
    """Raised when transcription fails"""
    pass

# Example usage:
def create_audio_processor(model_size: str) -> AudioProcessor:
    extractor = MoviePyAudioExtractor()
    transcriber = WhisperTranscriber(model_size)
    return AudioProcessor(extractor, transcriber)