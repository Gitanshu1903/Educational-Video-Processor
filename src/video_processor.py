from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
import tempfile
import os

@dataclass
class VideoSpec:
    """Video specifications and output settings"""
    fps: int
    codec: str
    audio_codec: str
    threads: int
    bitrate: str
    preset: str  # encoding preset (e.g., "medium", "fast")
    temp_directory: str

@dataclass
class ProcessingProgress:
    """Progress information for video processing"""
    total_frames: int
    processed_frames: int
    current_stage: str
    eta_seconds: float

class ProgressCallback(ABC):
    """Abstract base class for progress reporting"""
    @abstractmethod
    def on_progress(self, progress: ProcessingProgress) -> None:
        pass

class ConsoleProgressCallback(ProgressCallback):
    """Console-based progress reporting"""
    def on_progress(self, progress: ProcessingProgress) -> None:
        percentage = (progress.processed_frames / progress.total_frames) * 100
        print(f"Stage: {progress.current_stage} - "
              f"Progress: {percentage:.1f}% - "
              f"ETA: {progress.eta_seconds:.1f}s")

class VideoLoader(ABC):
    """Abstract base class for video loading strategies"""
    @abstractmethod
    def load(self, path: str) -> VideoFileClip:
        pass

class StandardVideoLoader(VideoLoader):
    """Standard implementation of video loading"""
    def load(self, path: str) -> VideoFileClip:
        if not os.path.exists(path):
            raise VideoLoadError(f"Video file not found: {path}")
        
        try:
            return VideoFileClip(path)
        except Exception as e:
            raise VideoLoadError(f"Failed to load video: {str(e)}")

class VideoWriter(ABC):
    """Abstract base class for video writing strategies"""
    @abstractmethod
    def write(self, 
             video: CompositeVideoClip, 
             output_path: str, 
             spec: VideoSpec,
             progress_callback: Optional[ProgressCallback] = None) -> None:
        pass

class OptimizedVideoWriter(VideoWriter):
    """Optimized implementation of video writing with threading"""
    def write(self, 
             video: CompositeVideoClip, 
             output_path: str, 
             spec: VideoSpec,
             progress_callback: Optional[ProgressCallback] = None) -> None:
        try:
            # Create temporary directory if it doesn't exist
            temp_dir = Path(spec.temp_directory)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary file for writing
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name

            # Write to temporary file first
            video.write_videofile(
                temp_path,
                fps=spec.fps,
                codec=spec.codec,
                audio_codec=spec.audio_codec,
                threads=spec.threads,
                bitrate=spec.bitrate,
                preset=spec.preset,
                logger=None,  # Use custom progress callback instead
                #progress_bar=False
            )

            # Move temporary file to final destination
            os.replace(temp_path, output_path)

        except Exception as e:
            # Clean up temporary file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise VideoWriteError(f"Failed to write video: {str(e)}")

class CaptionCompositor(ABC):
    """Abstract base class for caption composition strategies"""
    @abstractmethod
    def compose(self, 
                video: VideoFileClip, 
                captions: List[Dict[str, Any]],
                background: ColorClip) -> CompositeVideoClip:
        pass

class StandardCaptionCompositor(CaptionCompositor):
    """Standard implementation of caption composition with fixed duration handling"""
    def compose(self, 
                video: VideoFileClip, 
                captions: List[Dict[str, Any]],
                background: ColorClip) -> CompositeVideoClip:
        try:
            # Ensure video has duration
            if not hasattr(video, 'duration') or video.duration is None:
                video.duration = video.reader.duration
            
            # Set background position and duration
            background = (background
                        .set_position(('center', 'bottom'))
                        .set_duration(video.duration))
            
            # Create caption clips
            caption_clips = []
            for caption in captions:
                # Create clip for each caption
                caption_clip = caption.set_duration(
                    caption.end_time - caption.start_time
                ).set_start(caption.start_time)
                caption_clips.append(caption_clip)
            
            # Combine all clips
            all_clips = [video] + caption_clips
            
            # Create final composite with explicit duration
            final_composite = CompositeVideoClip(
                all_clips,
                size=video.size
            ).set_duration(video.duration)
            
            # Ensure audio is preserved
            if video.audio is not None:
                final_composite = final_composite.set_audio(video.audio)
            
            return final_composite

        except Exception as e:
            raise CompositionError(f"Failed to compose video with captions: {str(e)}")



class VideoProcessor:
    """Main coordinator for video processing with improved duration handling"""
    def __init__(self,
                 loader: VideoLoader,
                 writer: VideoWriter,
                 compositor: CaptionCompositor,
                 spec: VideoSpec):
        self.loader = loader
        self.writer = writer
        self.compositor = compositor
        self.spec = spec
        self.logger = logging.getLogger(__name__)

    def process_video(self,
                     input_path: str,
                     output_path: str,
                     captions: List[Dict[str, Any]],
                     background: ColorClip,
                     progress_callback: Optional[ProgressCallback] = None) -> None:
        """Process video with captions"""
        input_video = None
        final_video = None

        try:
            # Load input video
            self.logger.info(f"Loading video from {input_path}")
            input_video = self.loader.load(input_path)
            
            # Ensure input video has duration
            if not hasattr(input_video, 'duration') or input_video.duration is None:
                input_video.duration = input_video.reader.duration
            
            # Log video properties for debugging
            self.logger.info(f"Video duration: {input_video.duration}")
            self.logger.info(f"Video size: {input_video.size}")

            # Validate captions
            self._validate_captions(captions, input_video.duration)

            # Compose video with captions
            self.logger.info("Compositing captions")
            final_video = self.compositor.compose(input_video, captions, background)
            
            # Ensure final video has proper duration
            if not hasattr(final_video, 'duration') or final_video.duration is None:
                final_video.duration = input_video.duration

            # Write output video
            self.logger.info(f"Writing video to {output_path}")
            self.writer.write(final_video, output_path, self.spec, progress_callback)

        except (VideoLoadError, CompositionError, VideoWriteError) as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            raise
        finally:
            # Clean up resources
            if input_video:
                input_video.close()
            if final_video:
                final_video.close()

    def _validate_captions(self, captions: List[Dict[str, Any]], video_duration: float) -> None:
        """Validate caption timing against video duration"""
        for caption in captions:
            if not hasattr(caption, 'start_time') or not hasattr(caption, 'end_time'):
                raise CompositionError("Captions must have start_time and end_time attributes")
            if caption.end_time > video_duration:
                self.logger.warning(f"Caption end time {caption.end_time} exceeds video duration {video_duration}")
                caption.end_time = video_duration

# Custom exceptions
class VideoLoadError(Exception):
    """Raised when video loading fails"""
    pass

class VideoWriteError(Exception):
    """Raised when video writing fails"""
    pass

class CompositionError(Exception):
    """Raised when video composition fails"""
    pass
