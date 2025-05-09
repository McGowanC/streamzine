from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, HttpUrl, Field
import os
from dotenv import load_dotenv
import re

# Import from our new youtube_utils module
from .youtube_utils import fetch_youtube_video_data, VideoMetadata as YouTubeVideoMetadata # Renaming to avoid conflict

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

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

# Updated response model to include fetched YouTube data
class ProcessVideoResponse(BaseModel):
    message: str
    video_url_received: HttpUrl
    search_intent_received: str | None
    video_data: YouTubeVideoMetadata | None = None # This will hold title, duration, transcript
    # Later, this will also include: llm_summary, llm_table_of_contents, llm_article_html, etc.

# --- Utility Functions --- (Keep your validate_youtube_url function)
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
    return HealthCheckResponse(status="ok", message="API is healthy")

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

    video_metadata_result = await fetch_youtube_video_data(video_url_str)

    if video_metadata_result.error:
        # You could raise an HTTPException here if you prefer, 
        # or return it in the response for the client to handle.
        # For MVP, returning it in the response is okay.
        print(f"Error fetching video data: {video_metadata_result.error}")
        # We still return a 200 OK here, but with an error message in the video_data
        # Alternatively, if any error means failure, raise HTTPException:
        # raise HTTPException(status_code=422, detail=f"Failed to process video: {video_metadata_result.error}")
        return ProcessVideoResponse(
            message=f"Failed to fetch video data: {video_metadata_result.error}",
            video_url_received=request_data.video_url,
            search_intent_received=request_data.search_intent,
            video_data=video_metadata_result
        )

    # --- Placeholder for LLM processing logic ---
    # 1. If video_metadata_result.transcript is available:
    #    Call LLM to process transcript (Future Step)
    # 2. Format and return the article (Future Step)
    # --- End Placeholder ---
    
    # For now, return the fetched metadata and transcript.
    return ProcessVideoResponse(
        message="Successfully fetched video metadata and transcript.",
        video_url_received=request_data.video_url,
        search_intent_received=request_data.search_intent,
        video_data=video_metadata_result
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) # Note: "app.main:app" if running from backend_api dir