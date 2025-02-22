import streamlit as st
import tempfile
from pathlib import Path
import json
from typing import Optional
import logging
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
from summary_generator import TextSummaryGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('video_captioning')

def initialize_processors():
    """Initialize all necessary processors"""
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Initialize Audio Processor
    audio_processor = AudioProcessor(
        extractor=MoviePyAudioExtractor(),
        transcriber=WhisperTranscriber(model_size="medium")
    )
    
    # Initialize Caption Style and Constraints
    style = CaptionStyle(
        font_name="Helvetica",
        font_size=32,
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
    
    # Initialize Caption Generator
    caption_generator = CaptionGenerator(
        formatter=StandardCaptionFormatter(),
        clip_generator=StandardClipGenerator(),
        style=style,
        constraints=constraints
    )
    
    # Initialize Video Processor
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
    
    # Initialize Summary Generator
    summary_generator = TextSummaryGenerator()
    
    return (
        audio_processor,
        caption_generator,
        video_processor,
        summary_generator,
        style,
        temp_dir
    )

def process_video(
    uploaded_file,
    audio_processor,
    caption_generator,
    video_processor,
    summary_generator,
    style,
    temp_dir: Path
) -> Optional[tuple]:
    """Process the uploaded video file"""
    
    try:
        # Save uploaded file to temp directory
        input_path = temp_dir / "input_video.mp4"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process paths
        output_path = temp_dir / "output_video.mp4"
        summary_path = temp_dir / "summary.json"
        
        # Extract and transcribe audio
        word_level_info = audio_processor.process_video(str(input_path))
        
        # Generate summary
        summary_generator.generate_summary(word_level_info, summary_path)
        
        # Load video for frame size
        video = StandardVideoLoader().load(str(input_path))
        frame_size = video.size
        
        # Adjust font size based on video height
        style.font_size = int(frame_size[1] * 0.075)
        
        # Generate captions
        clips, background = caption_generator.generate_captions(
            word_level_info=word_level_info,
            frame_size=frame_size
        )
        video.close()
        
        # Set timing attributes
        for caption in clips:
            if not hasattr(caption, 'start_time'):
                caption.start_time = caption.start
            if not hasattr(caption, 'end_time'):
                caption.end_time = caption.end
        
        # Process final video
        video_processor.process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            captions=clips,
            background=background,
            progress_callback=ConsoleProgressCallback()
        )
        
        # Read summary
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        # Read processed video
        with open(output_path, 'rb') as f:
            processed_video = f.read()
            
        return processed_video, summary_data
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Educational Video Processor", layout="wide")
    
    st.title("Educational Video Processor")
    st.write("""
    Upload an educational video to:
    1. Add auto-generated captions
    2. Generate an AI summary
    3. Create word-level timestamps
    """)
    
    # Initialize processors
    processors = initialize_processors()
    audio_processor, caption_generator, video_processor, summary_generator, style, temp_dir = processors
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        with st.spinner("Processing video... This may take a few minutes."):
            result = process_video(
                uploaded_file,
                audio_processor,
                caption_generator,
                video_processor,
                summary_generator,
                style,
                temp_dir
            )
            
            if result:
                processed_video, summary_data = result
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Processed Video")
                    st.video(processed_video)
                    
                    # Download button for processed video
                    st.download_button(
                        label="Download Processed Video",
                        data=processed_video,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                
                with col2:
                    st.subheader("Video Summary")
                    
                    # Display key points
                    st.markdown("### Key Points")
                    st.write(summary_data["educational_summary"]["key_points"])
                    
                    # Display detailed summary
                    st.markdown("### Detailed Summary")
                    st.write(summary_data["educational_summary"]["detailed_summary"])
                    
                    # Video statistics
                    st.markdown("### Video Statistics")
                    st.write(f"Duration: {summary_data['duration_seconds']:.2f} seconds")
                    st.write(f"Total Words: {summary_data['total_words']}")
                    
                    # Download button for summary
                    st.download_button(
                        label="Download Summary (JSON)",
                        data=json.dumps(summary_data, indent=2),
                        file_name="video_summary.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()