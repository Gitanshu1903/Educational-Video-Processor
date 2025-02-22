# main.py
from pathlib import Path
import logging
from dataclasses import dataclass

# Import from our existing enhanced modules
from audio_processor import (
    AudioProcessor, MoviePyAudioExtractor, WhisperTranscriber,
    AudioExtractionError, TranscriptionError
)
from caption_generator import (
    CaptionGenerator, StandardCaptionFormatter, StandardClipGenerator,
    CaptionStyle, CaptionConstraints
)
from video_processor import (
    VideoProcessor, VideoSpec, ConsoleProgressCallback,
    StandardVideoLoader, OptimizedVideoWriter, StandardCaptionCompositor
)

from summary_generator import (
    TextSummaryGenerator
)

def setup_logger():
    """Set up logging configuration"""
    logger = logging.getLogger('video_captioning')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def main():
    # Set up logging
    logger = setup_logger()
    
    # Create necessary directories
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Initialize Audio Processor
        logger.info("Initializing audio processor...")
        audio_processor = AudioProcessor(
            extractor=MoviePyAudioExtractor(),
            transcriber=WhisperTranscriber(model_size="medium")
        )
        
        # 2. Initialize Caption Generator
        logger.info("Initializing caption generator...")
        style = CaptionStyle(
            font_name="Helvetica",
            font_size=32,  # Will be adjusted based on video height
            text_color="white",
            highlight_color="yellow",
            stroke_color="black",
            stroke_width=1.5,
            background_color=(64, 64, 64),
            background_opacity=0.6
        )
        
        constraints = CaptionConstraints(
            max_chars_per_line=30,
            max_duration_per_line=2.5,
            max_gap_between_words=1.5,
            x_buffer_ratio=0.1
        )
        
        caption_generator = CaptionGenerator(
            formatter=StandardCaptionFormatter(),
            clip_generator=StandardClipGenerator(),
            style=style,
            constraints=constraints
        )
        
        # 3. Initialize Video Processor
        logger.info("Initializing video processor...")
        spec = VideoSpec(
            fps=24,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            bitrate="8000k",
            preset="medium",
            temp_directory=str(temp_dir)
        )
        
        video_processor = VideoProcessor(
            loader=StandardVideoLoader(),
            writer=OptimizedVideoWriter(),
            compositor=StandardCaptionCompositor(),
            spec=spec
        )
        
        # 4. Process the video
        input_path = "videoplayback1.mp4"  # Replace with your input video path
        output_path = "output.mp4"  # Replace with your output video path
        
        logger.info("Starting video processing pipeline...")
        
        # Extract and transcribe audio
        logger.info("Processing audio...")
        word_level_info = audio_processor.process_video(input_path)
        
        # Generate captions
        logger.info("Generating captions...")
        video = StandardVideoLoader().load(input_path)
        frame_size = video.size
        
        # Adjust font size based on video height
        style.font_size = int(frame_size[1] * 0.075)  # 7.5% of video height
        
        clips, background = caption_generator.generate_captions(
            word_level_info=word_level_info,
            frame_size=frame_size
        )
        video.close()
        
        # Process final video
        logger.info("Creating final video...")

        # Ensure your captions have proper timing attributes
        for caption in clips:
            if not hasattr(caption, 'start_time'):
                caption.start_time = caption.start
            if not hasattr(caption, 'end_time'):
                caption.end_time = caption.end
        
        video_processor.process_video(
            input_path=input_path,
            output_path=output_path,
            captions=clips,
            background=background,
            progress_callback=ConsoleProgressCallback()
        )
        
        logger.info("Video processing completed successfully!")

        logger.info("Summarizing Video Content!")
        output_path = Path("summary.json")
        summary_generator = TextSummaryGenerator()
        summary_generator.generate_summary(word_level_info, output_path)


        
    except AudioExtractionError as e:
        logger.error(f"Audio extraction failed: {e}")
        return 1
    except TranscriptionError as e:
        logger.error(f"Transcription failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())