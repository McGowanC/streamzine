from pytube import YouTube
from pytube.exceptions import PytubeError, VideoUnavailable as PytubeVideoUnavailable # More specific Pytube errors
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable as TranscriptVideoUnavailable
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any

# Pydantic model for transcript segments (useful for typing)
class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float

class VideoMetadata(BaseModel):
    video_id: str
    title: str
    duration: int # in seconds
    transcript: List[TranscriptSegment] | None = None
    error: str | None = None # To store any errors during fetching
    error_type: str | None = None # To store the type of error

def get_video_id(url: str) -> str | None:
    """
    Extracts the YouTube video ID from a URL.
    Handles both youtube.com/watch?v=VIDEO_ID and youtu.be/VIDEO_ID formats.
    """
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

async def fetch_youtube_video_data(video_url_str: str) -> VideoMetadata:
    """
    Fetches YouTube video metadata (ID, title, duration) and its transcript.
    """
    video_id = get_video_id(video_url_str)
    title = "N/A" # Initialize default values
    duration = 0

    if not video_id:
        return VideoMetadata(video_id="", title="", duration=0, error="Invalid YouTube URL or could not extract Video ID.", error_type="InvalidURL")

    try:
        # Get Title and Duration using Pytube
        yt = YouTube(video_url_str)
        title = yt.title
        duration = yt.length # Duration in seconds

        # Get Transcript using youtube-transcript-api
        transcript_list_dict = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        
        transcript_segments = [
            TranscriptSegment(text=item['text'], start=item['start'], duration=item['duration'])
            for item in transcript_list_dict
        ]
        
        return VideoMetadata(
            video_id=video_id,
            title=title,
            duration=duration,
            transcript=transcript_segments
        )
    
    except PytubeVideoUnavailable as e_pv_unavailable:
        print(f"Pytube - Video unavailable for {video_id}: {e_pv_unavailable}")
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"Video is unavailable (pytube): {e_pv_unavailable}", error_type="PytubeVideoUnavailable")
    except PytubeError as e_pytube: # Catch general Pytube errors
        print(f"Pytube - An error occurred for {video_id}: {e_pytube}")
        # The error you saw "HTTP Error 400: Bad Request" would likely be caught here.
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"Pytube error: {e_pytube}", error_type="PytubeError")
    except TranscriptsDisabled:
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"Transcripts are disabled for video ID: {video_id}", error_type="TranscriptsDisabled")
    except NoTranscriptFound:
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"No English transcript found for video ID: {video_id}", error_type="NoTranscriptFound")
    except TranscriptVideoUnavailable: # This is from youtube_transcript_api
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"Video unavailable for transcript fetching: {video_id}", error_type="TranscriptVideoUnavailable")
    except Exception as e:
        # Catch other potential errors
        print(f"An unexpected error occurred while fetching video data for {video_id}: {e} (Type: {type(e).__name__})")
        return VideoMetadata(video_id=video_id, title=title, duration=duration, error=f"An unexpected error occurred: {str(e)}", error_type=type(e).__name__)