import torch
from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import pipeline
from typing import Optional, List, Dict


class HuggingFaceSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize summarizer with chosen model
        Options:
        - "facebook/bart-large-cnn" (default)
        - "google/pegasus-xsum"
        - "philschmid/bart-large-cnn-samsum"
        """
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

    @staticmethod
    def extract_video_id(youtube_url: str) -> Optional[str]:
        """Extract video ID from URL"""
        video_id_regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(video_id_regex, youtube_url)
        return match.group(1) if match else None

    @staticmethod
    def get_transcript(video_id: str) -> Optional[str]:
        """Get video transcript"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join(entry['text'] for entry in transcript)
        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            return None

    @staticmethod
    def chunk_text(text: str, max_length: int = 1024) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def summarize_text(self, text: str) -> Optional[str]:
        """Summarize text using HuggingFace pipeline"""
        try:
            chunks = self.chunk_text(text)
            summaries = []

            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])

            # # If multiple chunks, summarize the summaries
            # if len(summaries) > 1:
            #     final_summary = self.summarizer(
            #         ' '.join(summaries),
            #         max_length=400,
            #         min_length=30,
            #         do_sample=False
            #     )[0]['summary_text']
            #     return final_summary

            return summaries[0]

        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return None

    def process_video(self, youtube_url: str) -> Dict:
        """Process entire video"""
        result = {
            "video_id": None,
            "transcript": None,
            "summary": None
        }

        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            print("Invalid YouTube URL")
            return result
        result["video_id"] = video_id

        transcript = self.get_transcript(video_id)
        if not transcript:
            print("Could not get transcript")
            return result
        result["transcript"] = transcript

        summary = self.summarize_text(transcript)
        if not summary:
            print("Could not generate summary")
            return result
        result["summary"] = summary

        return result


def main():
    # Initialize summarizer
    summarizer = HuggingFaceSummarizer()

    # Example usage
    youtube_url = input("Enter YouTube video URL: ")
    result = summarizer.process_video(youtube_url)

    if result["summary"]:
        print("\nVideo Summary:")
        print("-" * 50)
        print(result["summary"])
    else:
        print("Failed to generate summary")


if __name__ == "__main__":
    main()