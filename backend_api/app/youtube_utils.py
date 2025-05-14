import asyncio
import functools
import yt_dlp
from yt_dlp.utils import DownloadError as YtDlpDownloadError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable as TranscriptVideoUnavailable
# Removed: from youtube_transcript_api._errors import RequestsError as YoutubeTranscriptApiRequestsError
# We will catch more general exceptions or rely on the library's specific ones.
# If needed, you could import requests.exceptions directly if you wanted to catch its errors
# import requests.exceptions # Example, not used in this immediate fix

from pydantic import BaseModel
from typing import List

# ... (TranscriptSegment and VideoMetadata Pydantic models remain the same) ...
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


def get_video_id_from_url(url: str) -> str | None: # Renamed to avoid conflict if you keep old pytube
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

async def _fetch_metadata_with_yt_dlp(video_url_str: str, attempt_num: int, max_attempts: int):
    print(f"yt-dlp: Attempt {attempt_num}/{max_attempts} fetching metadata for {video_url_str}")
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best',
        'skip_download': True,
        'simulate': True,
        'extract_flat': 'discard_in_playlist',
        'nocheckcertificate': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            blocking_call = functools.partial(ydl.extract_info, video_url_str, download=False)
            info_dict = await asyncio.to_thread(blocking_call)
        
        video_id_fetched = info_dict.get("id")
        title_fetched = info_dict.get("title", "N/A")
        duration_fetched = int(info_dict.get("duration", 0))
        print(f"yt-dlp: Success - Title: '{title_fetched}', Duration: {duration_fetched}s for video ID {video_id_fetched}")
        return video_id_fetched, title_fetched, duration_fetched, None, None
    except YtDlpDownloadError as e:
        error_message = f"yt-dlp DownloadError: {str(e)}"
        error_type = "YtDlpDownloadError"
        print(f"{error_type} for {video_url_str}: {error_message}")
        return None, "N/A", 0, error_message, error_type
    except Exception as e:
        error_message = f"Unexpected error in _fetch_metadata_with_yt_dlp: {str(e)}"
        error_type = type(e).__name__
        print(f"{error_type} for {video_url_str}: {error_message}")
        return None, "N/A", 0, error_message, error_type

async def _fetch_transcript_with_api(video_id: str, attempt_num: int, max_attempts: int):
    print(f"TranscriptAPI: Attempt {attempt_num}/{max_attempts} fetching transcript for {video_id}")
    try:
        blocking_call = functools.partial(YouTubeTranscriptApi.get_transcript, video_id, languages=['en', 'en-US'])
        transcript_list_dict = await asyncio.to_thread(blocking_call)
        transcript_segments = [
            TranscriptSegment(text=item['text'], start=item['start'], duration=item['duration'])
            for item in transcript_list_dict
        ]
        print(f"TranscriptAPI: Success - Fetched transcript for {video_id}")
        return transcript_segments, None, None
    except (TranscriptsDisabled, NoTranscriptFound, TranscriptVideoUnavailable) as e_transcript:
        error_message = f"Transcript Error: {str(e_transcript)}"
        error_type = type(e_transcript).__name__
        print(f"{error_type} for {video_id}: {error_message}")
        return None, error_message, error_type
    # Catching a more general Exception for other potential network issues from the underlying requests library
    # If youtube_transcript_api raises an error that isn't one of its specific ones above,
    # it will be caught here.
    except Exception as e:
        error_message = f"Error during transcript fetching (possibly network related): {str(e)}"
        error_type = f"TranscriptFetch_{type(e).__name__}" # More specific error type
        print(f"{error_type} for {video_id}: {error_message}")
        return None, error_message, error_type


# ... (The rest of fetch_youtube_video_data orchestrator function remains the same as the previous version I gave you) ...
async def fetch_youtube_video_data(video_url_str: str) -> VideoMetadata:
    # Get initial video_id from URL, mainly for error reporting if metadata fetch fails entirely
    parsed_video_id_from_url = get_video_id_from_url(video_url_str)
    if not parsed_video_id_from_url:
        return VideoMetadata(video_id="", title="N/A", duration=0, error="Invalid YouTube URL or could not extract Video ID.", error_type="InvalidURL")

    max_retries = 3
    base_backoff_seconds = 3 # Slightly increased base backoff

    # --- Stage 1: Fetch Metadata (Title, Duration, Confirmed Video ID) with yt-dlp ---
    video_id_final = parsed_video_id_from_url
    title_final = "N/A"
    duration_final = 0
    metadata_error_message = None
    metadata_error_type = None

    for attempt in range(max_retries):
        vid_id, title, duration, err_msg, err_type = await _fetch_metadata_with_yt_dlp(video_url_str, attempt + 1, max_retries)
        if vid_id and not err_msg: # Success
            video_id_final = vid_id
            title_final = title
            duration_final = duration
            metadata_error_message = None # Clear previous attempt errors
            metadata_error_type = None
            break 
        else: # Failure
            metadata_error_message = err_msg
            metadata_error_type = err_type
            # Check if error message from yt-dlp indicates a 429 or similar retryable http error
            if err_msg and ("429" in err_msg or "too many requests" in err_msg.lower() or "urlopen error http error 403" in err_msg.lower()):
                if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt)
                    print(f"Retryable error detected from yt-dlp for metadata. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for yt-dlp metadata fetch with error: {err_msg}")
                    break 
            else: # Non-retryable yt-dlp error or other unexpected error
                print(f"Non-retryable or unexpected error from yt-dlp metadata fetch: {err_msg}")
                break 
    
    if metadata_error_message: # Metadata fetching failed after all retries or with a non-retryable error
        return VideoMetadata(
            video_id=video_id_final, 
            title=title_final, 
            duration=duration_final, 
            transcript=None,
            error=metadata_error_message,
            error_type=metadata_error_type
        )

    # --- Stage 2: Fetch Transcript with youtube-transcript-api ---
    transcript_final = None
    transcript_error_message = None
    transcript_error_type = None

    for attempt in range(max_retries):
        transcript_segments, err_msg, err_type = await _fetch_transcript_with_api(video_id_final, attempt + 1, max_retries)
        if transcript_segments and not err_msg: # Success
            transcript_final = transcript_segments
            transcript_error_message = None # Clear previous attempt errors
            transcript_error_type = None
            break
        else: # Failure
            transcript_error_message = err_msg
            transcript_error_type = err_type
            # Check if error message from transcript API indicates a 429 or similar retryable http error
            if err_msg and ("429" in err_msg or "too many requests" in err_msg.lower() or "http error 403" in err_msg.lower()):
                if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt)
                    print(f"Retryable error detected from transcript API. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for transcript API fetch with error: {err_msg}")
                    break
            else: # Non-retryable transcript API error
                print(f"Non-retryable or unexpected error from transcript API: {err_msg}")
                break
    
    # Return combined results
    return VideoMetadata(
        video_id=video_id_final,
        title=title_final,
        duration=duration_final,
        transcript=transcript_final,
        error=transcript_error_message, # Report transcript error if it occurred, metadata was successful
        error_type=transcript_error_type
    )