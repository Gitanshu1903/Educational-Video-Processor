from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from moviepy.editor import TextClip, ColorClip
import logging

@dataclass
class CaptionStyle:
    """Configuration for caption styling"""
    font_name: str
    font_size: int
    text_color: str
    highlight_color: str
    stroke_color: str
    stroke_width: float
    background_color: Tuple[int, int, int]
    background_opacity: float

@dataclass
class CaptionConstraints:
    """Configuration for caption formatting constraints"""
    max_chars_per_line: int
    max_duration_per_line: float
    max_gap_between_words: float
    x_buffer_ratio: float = 0.1

@dataclass
class WordInfo:
    """Information about a single word in the caption"""
    text: str
    start_time: float
    end_time: float
    position: Optional[Tuple[float, float]] = None
    dimensions: Optional[Tuple[float, float]] = None

class CaptionLine:
    """Represents a line of caption text"""
    def __init__(self, words: List[WordInfo]):
        self.words = words
        self.start_time = min(word.start_time for word in words)
        self.end_time = max(word.end_time for word in words)
        self.duration = self.end_time - self.start_time

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

class CaptionFormatter(ABC):
    """Abstract base class for caption formatting strategies"""
    @abstractmethod
    def format_captions(self, words: List[WordInfo], constraints: CaptionConstraints) -> List[CaptionLine]:
        pass

class StandardCaptionFormatter(CaptionFormatter):
    """Standard implementation of caption formatting"""
    def format_captions(self, words: List[WordInfo], constraints: CaptionConstraints) -> List[CaptionLine]:
        lines: List[CaptionLine] = []
        current_line: List[WordInfo] = []
        current_duration = 0

        for i, word in enumerate(words):
            current_line.append(word)
            current_duration += word.end_time - word.start_time
            
            # Check if we should create a new line
            should_break = self._should_break_line(
                current_line, current_duration, words, i, constraints
            )

            if should_break and current_line:
                lines.append(CaptionLine(current_line))
                current_line = []
                current_duration = 0

        # Add remaining words
        if current_line:
            lines.append(CaptionLine(current_line))

        return lines

    def _should_break_line(
        self, 
        current_line: List[WordInfo],
        current_duration: float,
        all_words: List[WordInfo],
        current_index: int,
        constraints: CaptionConstraints
    ) -> bool:
        # Check character limit
        line_text = " ".join(word.text for word in current_line)
        if len(line_text) > constraints.max_chars_per_line:
            return True

        # Check duration limit
        if current_duration > constraints.max_duration_per_line:
            return True

        # Check gap between words
        if current_index > 0:
            gap = all_words[current_index].start_time - all_words[current_index - 1].end_time
            if gap > constraints.max_gap_between_words:
                return True

        return False

class ClipGenerator(ABC):
    """Abstract base class for generating video clips from captions"""
    @abstractmethod
    def create_clips(self, caption_line: CaptionLine, style: CaptionStyle, frame_size: Tuple[int, int]) -> List[TextClip]:
        pass

class StandardClipGenerator(ClipGenerator):
    """Standard implementation of clip generation"""
    def create_clips(
        self, 
        caption_line: CaptionLine, 
        style: CaptionStyle, 
        constraints: CaptionConstraints,
        frame_size: Tuple[int, int]
    ) -> List[TextClip]:
        clips = []
        frame_width, frame_height = frame_size
        
        try:
            # Define position buffers
            constraints.x_buffer_ratio = frame_width * 1/10
            max_width = frame_width - 2 * (constraints.x_buffer_ratio)

            # Start positioning from lower middle
            y_pos = frame_height * 0.75  # 75% from the top
            x_pos = (frame_width - max_width) / 2  # Center horizontally

            for word in caption_line.words:
                # Create normal and highlighted clips
                word_clip = self._create_word_clip(
                    word, style, caption_line.duration, caption_line.start_time,
                    x_pos, y_pos, style.text_color
                )
                
                highlight_clip = self._create_word_clip(
                    word, style, word.end_time - word.start_time, word.start_time,
                    x_pos, y_pos, style.highlight_color
                )

                # Update position based on word width
                clip_width, clip_height = word_clip.size
                if x_pos + clip_width > max_width:
                    x_pos = (frame_width - max_width) / 2  # Reset to center
                    y_pos += clip_height + 10  # Move to next line
                
                clips.extend([word_clip, highlight_clip])
                x_pos += clip_width + self._get_space_width(style)

        except Exception as e:
            logging.error(f"Error creating clips for caption line: {str(e)}")
            raise ClipGenerationError(f"Failed to create clips: {str(e)}")

        return clips


    def _create_word_clip(
        self, 
        word: WordInfo,
        style: CaptionStyle,
        duration: float,
        start_time: float,
        x_pos: float,
        y_pos: float,
        color: str
    ) -> TextClip:
        return (TextClip(
            word.text,
            font=style.font_name,
            fontsize=style.font_size,
            color=color,
            stroke_color=style.stroke_color,
            stroke_width=style.stroke_width
        )
        .set_start(start_time)
        .set_duration(duration)
        .set_position((x_pos, y_pos)))

    def _get_space_width(self, style: CaptionStyle) -> int:
        """Get width of a space character for the current style"""
        space_clip = TextClip(" ", font=style.font_name, fontsize=style.font_size)
        width = space_clip.size[0]
        space_clip.close()
        return width

class CaptionGenerator:
    """Main coordinator for caption generation process"""
    def __init__(
        self,
        formatter: CaptionFormatter,
        clip_generator: ClipGenerator,
        style: CaptionStyle,
        constraints: CaptionConstraints
    ):
        self.formatter = formatter
        self.clip_generator = clip_generator
        self.style = style
        self.constraints = constraints

    def generate_captions(
        self,
        word_level_info: List[Dict],
        frame_size: Tuple[int, int]
    ) -> Tuple[List[TextClip], ColorClip]:
        try:
            # Convert word info to WordInfo objects
            words = [
                WordInfo(
                    text=info['word'],
                    start_time=info['start'],
                    end_time=info['end']
                )
                for info in word_level_info
            ]

            # Format into lines
            caption_lines = self.formatter.format_captions(words, self.constraints)

            # Generate clips for each line
            all_clips = []
            for line in caption_lines:
                clips = self.clip_generator.create_clips(line, self.style, self.constraints ,frame_size)
                all_clips.extend(clips)

            # Create background
            background = self._create_background(all_clips, frame_size)

            return all_clips, background

        except Exception as e:
            logging.error(f"Caption generation failed: {str(e)}")
            raise CaptionGenerationError(f"Failed to generate captions: {str(e)}")

    def _create_background(
        self,
        clips: List[TextClip],
        frame_size: Tuple[int, int]
    ) -> ColorClip:
        """Create semi-transparent background for captions"""
        max_width = max(clip.size[0] for clip in clips)
        max_height = max(clip.size[1] for clip in clips)
        
        return (ColorClip(
            size=(int(max_width * 1.1), int(max_height * 1.1)),
            color=self.style.background_color
        )
        .set_opacity(self.style.background_opacity))

# Custom exceptions
class ClipGenerationError(Exception):
    """Raised when clip generation fails"""
    pass

class CaptionGenerationError(Exception):
    """Raised when caption generation fails"""
    pass
