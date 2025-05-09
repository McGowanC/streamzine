from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, HttpUrl, Field
import os
from dotenv import load_dotenv
import re
from typing import List, Dict, Any # Added for type hinting

# Load environment variables from .env file if it exists (for local development)
load_dotenv()
print(f"DEBUG: ANTHROPIC_API_KEY in main.py after load_dotenv: {os.getenv('ANTHROPIC_API_KEY')}") # Debug line

# Import from our utility modules
from .youtube_utils import fetch_youtube_video_data, VideoMetadata as YouTubeVideoMetadata, TranscriptSegment
from .llm_service import generate_article_from_transcript, LLMArticleOutput, ANTHROPIC_API_KEY_AVAILABLE


app = FastAPI(
    title="Video-to-Article AI API",
    description="API to process YouTube videos into structured articles.",
    version="0.1.0",
)

# --- Pydantic Models (Request/Response Schemas) ---

class HealthCheckResponse(BaseModel):
    status: str
    message: str

class ProcessVideoRequest(BaseModel):
    video_url: HttpUrl
    search_intent: str | None = Field(
        default=None,
        title="User's Search Intent",
        description="Optional text describing what the user is looking for in the video.",
        max_length=500
    )

# Updated response model to include LLM processed data
class ProcessVideoResponse(BaseModel):
    message: str
    video_url_received: HttpUrl
    search_intent_received: str | None
    video_data: YouTubeVideoMetadata | None = None
    llm_article_data: LLMArticleOutput | None = None # To hold summary, ToC, article sections

# --- Utility Functions ---
YOUTUBE_URL_PATTERN = re.compile(
    r"^(https://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})$"
)

def validate_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_PATTERN.match(url))

# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Video-to-Article AI API!"}

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
async def health_check():
    # Check if LLM API Key is available as part of health
    if not ANTHROPIC_API_KEY_AVAILABLE:
        return HealthCheckResponse(status="degraded", message="API is running, but LLM service may be unavailable due to missing API key.")
    return HealthCheckResponse(status="ok", message="API is healthy and LLM client initialized.")

@app.post("/process-video", response_model=ProcessVideoResponse, tags=["Video Processing"])
async def process_video_endpoint(request_data: ProcessVideoRequest):
    video_url_str = str(request_data.video_url)

    if not validate_youtube_url(video_url_str):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube video URL format. Please use a valid youtube.com/watch or youtu.be URL."
        )

    print(f"Processing video URL: {video_url_str}")
    if request_data.search_intent:
        print(f"Search intent: {request_data.search_intent}")

    # 1. Fetch YouTube video data (metadata and transcript)
    video_metadata_result = await fetch_youtube_video_data(video_url_str)

    if video_metadata_result.error:
        print(f"Error fetching video data: {video_metadata_result.error}")
        # Return error from YouTube fetching stage
        return ProcessVideoResponse(
            message=f"Failed to fetch video data: {video_metadata_result.error}",
            video_url_received=request_data.video_url,
            search_intent_received=request_data.search_intent,
            video_data=video_metadata_result, # Include the partial data with the error
            llm_article_data=None
        )

    if not video_metadata_result.transcript or not video_metadata_result.transcript:
        print("No transcript available to process with LLM.")
        return ProcessVideoResponse(
            message="Successfully fetched video metadata, but no transcript was available for LLM processing.",
            video_url_received=request_data.video_url,
            search_intent_received=request_data.search_intent,
            video_data=video_metadata_result,
            llm_article_data=None
        )

    # 2. If transcript is available, process with LLM
    print("Transcript fetched. Proceeding to LLM processing...")
    
    # Convert Pydantic TranscriptSegment objects to simple dicts if that's what llm_service expects
    # (Our llm_service currently expects List[Dict[str, Any]])
    transcript_for_llm: List[Dict[str, Any]] = []
    if video_metadata_result.transcript: # Should be true if we passed the check above
        for segment in video_metadata_result.transcript:
            transcript_for_llm.append({"text": segment.text, "start": segment.start, "duration": segment.duration})
    
    llm_processed_article: LLMArticleOutput | None = None
    if ANTHROPIC_API_KEY_AVAILABLE:
        try:
            llm_processed_article = await generate_article_from_transcript(
                transcript_segments=transcript_for_llm,
                user_search_intent=request_data.search_intent,
                video_title=video_metadata_result.title
            )
        except Exception as e:
            # Catch potential errors from the LLM service call itself if they weren't handled within
            print(f"Error during LLM processing: {e}")
            # Decide if you want to raise HTTPException or return an error in the response
            return ProcessVideoResponse(
                message=f"Successfully fetched video data, but LLM processing failed: {str(e)}",
                video_url_received=request_data.video_url,
                search_intent_received=request_data.search_intent,
                video_data=video_metadata_result,
                llm_article_data=LLMArticleOutput(summary=f"Error: LLM processing failed. {str(e)}", table_of_contents=[], article_sections=[]) # Provide a default error structure
            )
    else:
        print("LLM processing skipped due to API key unavailability.")
        # Optionally provide a specific LLMArticleOutput indicating skipped processing
        llm_processed_article = LLMArticleOutput(
            summary="LLM processing was skipped because the API key is not available.",
            table_of_contents=[],
            article_sections=[]
        )


    return ProcessVideoResponse(
        message="Video processed successfully.",
        video_url_received=request_data.video_url,
        search_intent_received=request_data.search_intent,
        video_data=video_metadata_result,
        llm_article_data=llm_processed_article
    )

if __name__ == "__main__":
    import uvicorn
    # Ensure Uvicorn uses the correct app instance string: "module_name:app_instance_name"
    # If main.py is in app/, and you run from backend_api/, it's "app.main:app"
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)