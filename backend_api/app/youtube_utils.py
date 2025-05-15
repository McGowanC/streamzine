import asyncio
import functools
import os # For getting API key from environment

# Google API Client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError as GoogleHttpError

# Transcript API (still needed)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable as TranscriptVideoUnavailable

from pydantic import BaseModel
from typing import List, Tuple, Optional # Added Tuple, Optional

# --- Pydantic Models (TranscriptSegment, VideoMetadata) remain the same ---
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

# --- Helper to parse ISO 8601 duration (from YouTube API) to seconds ---
def parse_iso8601_duration(duration_str: Optional[str]) -> int:
    if not duration_str or not duration_str.startswith('PT'):
        return 0
    
    duration_str = duration_str[2:] # Remove 'PT'
    seconds = 0
    minutes = 0
    hours = 0

    if 'H' in duration_str:
        parts = duration_str.split('H')
        hours = int(parts[0])
        duration_str = parts[1] if len(parts) > 1 else ''
    if 'M' in duration_str:
        parts = duration_str.split('M')
        minutes = int(parts[0])
        duration_str = parts[1] if len(parts) > 1 else ''
    if 'S' in duration_str:
        seconds = int(duration_str.replace('S', ''))
    
    return hours * 3600 + minutes * 60 + seconds

# --- Function to get video ID (remains the same) ---
def get_video_id_from_url(url: str) -> str | None:
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# --- NEW: Fetch metadata using YouTube Data API v3 ---
async def _fetch_metadata_with_youtube_api(
    video_id: str, 
    api_key: str, 
    attempt_num: int, 
    max_attempts: int
) -> Tuple[Optional[str], int, Optional[str], Optional[str]]:
    # Returns: title, duration_seconds, error_message, error_type
    print(f"YouTubeAPI: Attempt {attempt_num}/{max_attempts} fetching metadata for {video_id}")
    try:
        # serviceName, version, developerKey
        # This is a blocking call, so run in a thread
        youtube_service = await asyncio.to_thread(build, 'youtube', 'v3', developerKey=api_key)
        
        request = youtube_service.videos().list(
            part="snippet,contentDetails", # snippet for title, contentDetails for duration
            id=video_id
        )
        # This is also blocking
        response = await asyncio.to_thread(request.execute)

        if not response.get("items"):
            print(f"YouTubeAPI: Video not found or no items returned for {video_id}")
            return "N/A", 0, "Video not found via API", "YouTubeAPIVideoNotFound"
        
        video_item = response["items"][0]
        title = video_item["snippet"]["title"]
        iso_duration = video_item["contentDetails"].get("duration")
        duration_seconds = parse_iso8601_duration(iso_duration)
        
        print(f"YouTubeAPI: Success - Title: '{title}', Duration: {duration_seconds}s for {video_id}")
        return title, duration_seconds, None, None # Success
        
    except GoogleHttpError as e:
        error_content = e.resp.reason if hasattr(e.resp, 'reason') else str(e)
        try: # Try to parse error details from content
            error_details = e.content.decode()
            error_content = f"{error_content} - Details: {error_details}"
        except: pass

        error_message = f"YouTube Data API HttpError: {e.status_code} {error_content}"
        error_type = f"YouTubeAPI_HttpError_{e.status_code}"
        print(f"{error_type} for {video_id}: {error_message}")
        # Specific check for quotaExceeded or similar errors
        if "quotaExceeded" in error_message or "servingLimitExceeded" in error_message:
            error_type = "YouTubeAPI_QuotaExceeded"
        return "N/A", 0, error_message, error_type
    except Exception as e:
        error_message = f"Unexpected error fetching metadata via YouTube API: {str(e)}"
        error_type = f"YouTubeAPI_{type(e).__name__}"
        print(f"{error_type} for {video_id}: {error_message}")
        return "N/A", 0, error_message, error_type

# --- Fetch transcript (remains largely the same, but uses asyncio.to_thread) ---
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
    except Exception as e:
        error_message = f"Error during transcript fetching: {str(e)}"
        error_type = f"TranscriptFetch_{type(e).__name__}"
        print(f"{error_type} for {video_id}: {error_message}")
        return None, error_message, error_type

# --- Main Orchestrator Function (Modified) ---
async def fetch_youtube_video_data(video_url_str: str) -> VideoMetadata:
    youtube_api_key = os.getenv("YOUTUBE_DATA_API_KEY")
    if not youtube_api_key:
        print("ERROR: YOUTUBE_DATA_API_KEY environment variable not set.")
        return VideoMetadata(video_id="", title="", duration=0, error="Server configuration error: Missing YouTube Data API Key.", error_type="ServerConfigError")

    video_id_from_url = get_video_id_from_url(video_url_str)
    if not video_id_from_url:
        return VideoMetadata(video_id="", title="N/A", duration=0, error="Invalid YouTube URL or could not extract Video ID.", error_type="InvalidURL")

    max_retries = 3
    base_backoff_seconds = 2

    # --- Stage 1: Fetch Metadata with YouTube Data API v3 ---
    title_final = "N/A"
    duration_final = 0
    metadata_error_message: Optional[str] = None
    metadata_error_type: Optional[str] = None

    for attempt in range(max_retries):
        title, duration_s, err_msg, err_type = await _fetch_metadata_with_youtube_api(
            video_id_from_url, youtube_api_key, attempt + 1, max_retries
        )
        if not err_msg and title != "N/A": # Success if no error and title is found
            title_final = title
            duration_final = duration_s
            metadata_error_message = None
            metadata_error_type = None
            break
        else:
            metadata_error_message = err_msg
            metadata_error_type = err_type
            # Retry for specific API errors like 403 (Forbidden/Quota), 500, 503 (Server errors)
            # The GoogleHttpError parsing above will set error_type like "YouTubeAPI_HttpError_403"
            is_retryable_api_error = err_type and ("HttpError_403" in err_type or "HttpError_500" in err_type or "HttpError_503" in err_type or "QuotaExceeded" in err_type)
            if is_retryable_api_error:
                if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt)
                    print(f"Retryable YouTube Data API error ({err_type}). Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for YouTube Data API metadata fetch with error: {err_msg}")
                    break
            else: # Non-retryable API error
                print(f"Non-retryable error from YouTube Data API metadata fetch: {err_msg}")
                break
    
    if metadata_error_message:
        return VideoMetadata(
            video_id=video_id_from_url,
            title=title_final, # Might be "N/A" if error occurred on first try
            duration=duration_final, # Might be 0
            transcript=None,
            error=metadata_error_message,
            error_type=metadata_error_type
        )

    # --- Stage 2: Fetch Transcript (if metadata was successful) ---
    transcript_final = None
    transcript_error_message: Optional[str] = None
    transcript_error_type: Optional[str] = None

    # (Using the same retry logic for transcript API as before)
    for attempt in range(max_retries):
        transcript_segments, err_msg, err_type = await _fetch_transcript_with_api(video_id_from_url, attempt + 1, max_retries)
        if transcript_segments and not err_msg: # Success
            transcript_final = transcript_segments
            transcript_error_message = None 
            transcript_error_type = None
            break
        else: # Failure
            transcript_error_message = err_msg
            transcript_error_type = err_type
            if err_msg and ("429" in err_msg or "too many requests" in err_msg.lower() or "http error 403" in err_msg.lower()):
                if attempt < max_retries - 1:
                    wait_time = base_backoff_seconds * (2 ** attempt)
                    print(f"Retryable error detected from transcript API. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Max retries reached for transcript API fetch with error: {err_msg}")
                    break
            else: 
                print(f"Non-retryable or unexpected error from transcript API: {err_msg}")
                break
    
    return VideoMetadata(
        video_id=video_id_from_url,
        title=title_final,
        duration=duration_final,
        transcript=transcript_final,
        error=transcript_error_message, # Report transcript error if it occurred
        error_type=transcript_error_type
    )