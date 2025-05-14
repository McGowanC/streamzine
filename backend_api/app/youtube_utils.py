import asyncio # Added for asyncio.sleep()
from pytube import YouTube
from pytube.exceptions import PytubeError, VideoUnavailable as PytubeVideoUnavailable
from urllib.error import HTTPError as UrllibHTTPError # For catching HTTPError from pytube's underlying requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable as TranscriptVideoUnavailable
from pydantic import BaseModel
from typing import List # Removed Dict, Any as they weren't used in this file's models

# Pydantic model for transcript segments
class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float

class VideoMetadata(BaseModel):
    video_id: str
    title: str
    duration: int # in seconds
    transcript: List[TranscriptSegment] | None = None
    error: str | None = None
    error_type: str | None = None

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
    Fetches YouTube video metadata (ID, title, duration) and its transcript,
    with retry logic for 429 errors.
    """
    video_id = get_video_id(video_url_str)
    # Initialize defaults, these might be partially filled if only transcript fails
    title_fetched = "N/A"
    duration_fetched = 0
    
    # Variables for retry logic
    max_retries = 3
    base_backoff_seconds = 2 # Start with 2 seconds

    # To store the final error if all retries fail
    final_error_message: str | None = None
    final_error_type: str | None = None

    if not video_id:
        return VideoMetadata(video_id="", title="", duration=0, error="Invalid YouTube URL or could not extract Video ID.", error_type="InvalidURL")

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to fetch video data for {video_id} (URL: {video_url_str})")
            
            # --- Fetch Title and Duration using Pytube ---
            print(f"Pytube: Fetching metadata for {video_id}...")
            yt = YouTube(video_url_str)
            title_fetched = yt.title
            duration_fetched = yt.length
            print(f"Pytube: Success - Title: '{title_fetched}', Duration: {duration_fetched}s for {video_id}")

            # --- Get Transcript using youtube-transcript-api ---
            print(f"TranscriptAPI: Fetching transcript for {video_id}...")
            transcript_list_dict = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            transcript_segments = [
                TranscriptSegment(text=item['text'], start=item['start'], duration=item['duration'])
                for item in transcript_list_dict
            ]
            print(f"TranscriptAPI: Success - Fetched transcript for {video_id}")
            
            # If both succeeded
            return VideoMetadata(
                video_id=video_id,
                title=title_fetched,
                duration=duration_fetched,
                transcript=transcript_segments
            )
        
        except UrllibHTTPError as e_http:
            final_error_message = f"HTTP Error during YouTube fetch: {e_http.code} {e_http.reason}"
            final_error_type = f"HTTPError_{e_http.code}"
            print(f"Encountered {final_error_type} for {video_id}: {final_error_message}")
            if e_http.code == 429: # Too Many Requests
                if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt) # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time) # Use asyncio.sleep for async functions
                    continue # Go to the next attempt
                else:
                    print(f"Max retries reached for 429 error on {video_id}.")
                    # Error message already set, loop will terminate
            else: # Other HTTPErrors are not retried by this logic
                print(f"Non-429 HTTPError for {video_id}. Not retrying this error.")
            break # Break from retry loop for non-429 HTTPError or max retries for 429

        except (TranscriptsDisabled, NoTranscriptFound, TranscriptVideoUnavailable) as e_transcript:
            # These errors are specific to transcript fetching. Metadata might have been fetched.
            # Not typically retryable in the context of 429s, but means transcript part failed.
            final_error_message = f"Transcript Error: {str(e_transcript)}"
            final_error_type = type(e_transcript).__name__
            print(f"{final_error_type} for {video_id}: {final_error_message}")
            # Return with metadata if available, but indicate transcript error
            return VideoMetadata(
                video_id=video_id, 
                title=title_fetched, 
                duration=duration_fetched, 
                transcript=None, 
                error=final_error_message, 
                error_type=final_error_type
            )

        except PytubeVideoUnavailable as e_pv_unavailable:
            final_error_message = f"Video is unavailable (pytube): {e_pv_unavailable}"
            final_error_type = "PytubeVideoUnavailable"
            print(f"{final_error_type} for {video_id}: {final_error_message}")
            break 
        except PytubeError as e_pytube:
            final_error_message = f"Pytube error: {e_pytube}"
            final_error_type = "PytubeError"
            print(f"{final_error_type} for {video_id}: {final_error_message}")
            # This could be a 429 manifesting as a generic PytubeError if not caught by UrllibHTTPError
            # Check if error string contains "429" or "Too Many Requests"
            if "429" in str(e_pytube).lower() or "too many requests" in str(e_pytube).lower():
                 if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt)
                    print(f"Potential 429 in PytubeError. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
            break
        except Exception as e:
            final_error_message = f"An unexpected error occurred: {str(e)}"
            final_error_type = type(e).__name__
            print(f"{final_error_type} for {video_id} on attempt {attempt + 1}: {final_error_message}")
            if attempt < max_retries - 1: # General retry for other unexpected errors too
                wait_time = base_backoff_seconds * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"Max retries reached for unexpected error on {video_id}.")
            break # Break from retry loop
            
    # If the loop completes without returning successfully (i.e., all retries failed or an error broke the loop)
    print(f"All attempts failed for {video_id}. Returning error: {final_error_type} - {final_error_message}")
    return VideoMetadata(
        video_id=video_id, 
        title=title_fetched, # title/duration might have been fetched in an attempt before the final error
        duration=duration_fetched, 
        transcript=None, 
        error=final_error_message or "Max retries reached or an unhandled error occurred.", 
        error_type=final_error_type or "UnknownRetryFailure"
    )