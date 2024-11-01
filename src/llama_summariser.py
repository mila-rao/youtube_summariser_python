from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List, Dict


class LlamaYouTubeSummarizer:
    def __init__(self):
        """
        Initialize with full model

        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use original Llama 2 (requires approval)
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
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

    def chunk_text(self, text: str, max_length: int = 2048) -> List[str]:
        """Split text into manageable chunks"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_tokens = len(self.tokenizer.encode(word))
            if current_length + word_tokens > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def summarize_chunk(self, text: str) -> str:
        """Summarize a single chunk of text"""
        prompt = f"""Please provide a concise summary of the following transcript:

            {text}

            Summary:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.replace(prompt, "").strip()

    def summarize_text(self, text: str) -> Optional[str]:
        """Summarize entire text"""
        try:
            chunks = self.chunk_text(text)
            summaries = []

            for chunk in chunks:
                summary = self.summarize_chunk(chunk)
                summaries.append(summary)

            if len(summaries) > 1:
                final_text = "\n".join(summaries)
                return final_text

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
    summarizer = LlamaYouTubeSummarizer()

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