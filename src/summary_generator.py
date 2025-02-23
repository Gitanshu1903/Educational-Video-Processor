from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Dict
from transformers import pipeline
import torch
import textwrap

@dataclass
class TextSummaryGenerator:
    """Generates text summary files from word-level transcription data using BART model"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summary generator with a pretrained model
        
        Args:
            model_name: Name of the pretrained model to use
        """
        # Initialize the summarization pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=device
        )
        
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        """
        Split text into chunks that won't exceed the model's max input length
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        return textwrap.wrap(text, max_chunk_size, break_long_words=False)
    
    def generate_educational_summary(self, text: str) -> Dict[str, str]:
        """
        Generate an educational summary including key points and main ideas
        
        Args:
            text: Input text to summarize
            
        Returns:
            Dictionary containing different types of summaries
        """
        # Split text into chunks if it's too long
        chunks = self._chunk_text(text)
        chunk_summaries = []
        
        # Process each chunk
        for chunk in chunks:
            summary = self.summarizer(
                chunk,
                max_length=150,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # Generate a shorter summary for key points
        key_points = self.summarizer(
            combined_summary,
            max_length=100,
            min_length=20,
            do_sample=False
        )[0]['summary_text']
        
        return {
            "detailed_summary": combined_summary,
            "key_points": key_points
        }

    def generate_summary(
        self,
        word_level_info: List[Dict],
        output_path: Path
    ) -> None:
        """
        Generate a text summary file from word-level transcription data
        
        Args:
            word_level_info: List of dictionaries containing word timing information
            output_path: Path where the summary file will be saved
        """
        # Combine words into full text
        full_text = " ".join(word["text"] for word in word_level_info)
        
        # Generate educational summary
        educational_summary = self.generate_educational_summary(full_text)
        
        # Create summary data
        summary_data = {
          #  "total_words": len(word_level_info),
          #  "duration_seconds": word_level_info[-1]["end"] - word_level_info[0]["start"],
            "full_transcript": full_text,
            "educational_summary": {
                "detailed_summary": educational_summary["detailed_summary"],
                "key_points": educational_summary["key_points"]
            },
          #  "word_timings": [
            #     {
            #         "word": word["text"],
            #         "start": word["start"],
            #         "end": word["end"],
            #         "confidence": word.get("confidence", None)
            #     }
            #     for word in word_level_info
            # ]
        }
        
        # Write summary to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)