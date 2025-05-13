import os
import anthropic
from pydantic import BaseModel
from typing import List, Dict, Any
import json

# Initialize the Anthropic client
# ... (rest of the initialization code remains the same) ...
try:
    client = anthropic.Anthropic()
    ANTHROPIC_API_KEY_AVAILABLE = True
    print("LLM Service: Anthropic client initialized successfully.")
except anthropic.APIConnectionError as e:
    print(f"LLM Service: Failed to connect to Anthropic API: {e}")
    client = None
    ANTHROPIC_API_KEY_AVAILABLE = False
except anthropic.AuthenticationError as e:
    print(f"LLM Service: Anthropic API Authentication Error: {e} - Make sure ANTHROPIC_API_KEY is set correctly.")
    client = None
    ANTHROPIC_API_KEY_AVAILABLE = False
except Exception as e:
    print(f"LLM Service: Error initializing Anthropic client: {e} (Type: {type(e).__name__})")
    client = None
    ANTHROPIC_API_KEY_AVAILABLE = False


DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-3-haiku-20240307")

class LLMArticleOutput(BaseModel):
    summary: str
    table_of_contents: List[str]
    article_sections: List[Dict[str, Any]] # Will now expect "content_block" and "timestamp_seconds"

async def generate_article_from_transcript(
    transcript_segments: List[Dict[str, Any]],
    user_search_intent: str | None,
    video_title: str,
    model_name: str = DEFAULT_LLM_MODEL
) -> LLMArticleOutput:
    """
    Generates a structured article from video transcript segments using an LLM.
    """
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        # ... (error handling for client init remains the same) ...
        print("LLM Service: Anthropic client not available or API key issue.")
        return LLMArticleOutput(
            summary="Error: LLM service not available or API key issue.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content_block": "LLM processing failed due to configuration.", "timestamp_seconds": None, "relevant_to_search_intent": False}]
        )


    if not transcript_segments:
        # ... (error handling for empty transcript remains the same) ...
        print("LLM Service: Transcript segments list is empty, cannot generate article.")
        return LLMArticleOutput(
            summary="Transcript was empty.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content_block": "Transcript was empty.", "timestamp_seconds": None, "relevant_to_search_intent": False}]
        )


    full_transcript_text = " ".join([segment.get('text', '') for segment in transcript_segments])
    if not full_transcript_text.strip():
        # ... (error handling for effectively empty transcript remains the same) ...
        print("LLM Service: Combined transcript text is empty or only whitespace, cannot generate article.")
        return LLMArticleOutput(
            summary="Transcript text was effectively empty after combining segments.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content_block": "Transcript text was empty.", "timestamp_seconds": None, "relevant_to_search_intent": False}]
        )

    system_prompt = """You are an expert summarizer and article writer. Your task is to transform a raw video transcript into a well-structured, engaging article.
The transcript segments you will process are provided with a start time in TOTAL SECONDS, like "At 123.45s: ...text...".

The article should include:
1.  A concise overall summary of the video content as a single string.
2.  A table of contents consisting of a list of strings, where each string is a main section heading.
3.  The main article content, as a list of objects, where each object represents a logical block of information or paragraph and has the following keys:
    - "heading": (string) The section heading this content block belongs to.
    - "content_block": (string) The textual content for this specific block or paragraph. THIS TEXT SHOULD NOT CONTAIN ANY EMBEDDED TIMESTAMPS.
    - "timestamp_seconds": (float or null) The single most relevant start time in TOTAL SECONDS (e.g., 123.45 or 664.0) from the transcript that corresponds to the beginning of this 'content_block'. If no specific timestamp is directly applicable to this block (e.g., it's a very general statement or introduction), use null for this value.
    - "relevant_to_search_intent": (boolean) True if this content_block is particularly relevant to the user's search intent (if one was provided), false otherwise. If no user_search_intent is provided, this should always be false.

Your goal is to break down the video's content into meaningful `content_block`s, each associated with its primary `timestamp_seconds`.
Ensure the article flows well. Act like you are the creator of the YouTube video. Rephrase spoken language into more formal written language where appropriate.
If a user_search_intent is provided, pay close attention to it.

Output the result STRICTLY as a single JSON object with the exact keys: "summary", "table_of_contents", and "article_sections".
Each item in "article_sections" must have "heading", "content_block", "timestamp_seconds" (float or null), and "relevant_to_search_intent".
"""

    user_message_content = f"Video Title: \"{video_title}\"\n\n"
    user_message_content += "Transcript with timestamps (start times are in TOTAL SECONDS):\n"

    max_segments_to_send = 200
    # ... (transcript truncation logic remains the same) ...
    segments_to_send = transcript_segments
    if len(transcript_segments) > max_segments_to_send:
        segments_to_send = transcript_segments[:max_segments_to_send]
        print(f"LLM Service: Transcript too long ({len(transcript_segments)} segments). Truncating to first {max_segments_to_send} segments for LLM.")

    for segment in segments_to_send:
        start_time = segment.get('start', 0.0)
        text = segment.get('text', '')
        user_message_content += f"- At {start_time:.2f}s: {text}\n"

    if len(transcript_segments) > len(segments_to_send):
        user_message_content += f"\n... (transcript truncated after {len(segments_to_send)} segments from a total of {len(transcript_segments)} segments due to length constraints) ..."

    user_message_content += f"\n\nPlease transform this transcript into an article based on the system instructions. For each item in 'article_sections', provide the 'content_block' as clean text, and a separate 'timestamp_seconds' field with the relevant start time in total seconds (e.g., 123.45) or null."
    # ... (search intent logic remains the same) ...
    if user_search_intent:
        user_message_content += f"\nThe user's search intent is: \"{user_search_intent}\". Please identify sections relevant to this intent and set 'relevant_to_search_intent' to true for those sections."
    else:
        user_message_content += "\nNo specific user search intent was provided, so 'relevant_to_search_intent' should be false for all sections."


    user_message_content += "\n\nRemember to provide your output STRICTLY as a single JSON object with keys: 'summary', 'table_of_contents', and 'article_sections'. Each item in 'article_sections' must have 'heading', 'content_block', 'timestamp_seconds' (float or null), and 'relevant_to_search_intent'."

    print("\n--- Sending to LLM ---")

    try:
        # ... (API call and initial response handling remain the same) ...
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message_content}
            ]
        )
        if response.content and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
            llm_output_text = response.content[0].text
            # ... (JSON cleaning logic remains the same) ...
            print("\n--- LLM Raw Output (first 500 chars) ---")
            print(llm_output_text[:500])
            print("--- End LLM Raw Output ---")

            cleaned_llm_output_text = llm_output_text.strip()
            if cleaned_llm_output_text.startswith("```json"):
                cleaned_llm_output_text = cleaned_llm_output_text[7:]
                if cleaned_llm_output_text.endswith("```"):
                    cleaned_llm_output_text = cleaned_llm_output_text[:-3]
            elif cleaned_llm_output_text.startswith("```"):
                cleaned_llm_output_text = cleaned_llm_output_text[3:]
                if cleaned_llm_output_text.endswith("```"):
                    cleaned_llm_output_text = cleaned_llm_output_text[:-3]
            cleaned_llm_output_text = cleaned_llm_output_text.strip()


            try:
                parsed_json = json.loads(cleaned_llm_output_text)

                raw_summary = parsed_json.get("summary", "Summary not provided.")
                # ... (summary and toc parsing remain similar) ...
                final_summary = "Summary not provided or in unexpected format."
                if isinstance(raw_summary, str):
                    final_summary = raw_summary
                elif isinstance(raw_summary, list):
                    final_summary = "\n".join(filter(None, [str(item) for item in raw_summary]))

                raw_toc = parsed_json.get("table_of_contents", [])
                final_toc = []
                if isinstance(raw_toc, list):
                    final_toc = [str(item) for item in raw_toc if isinstance(item, str)]
                else:
                    print(f"Warning: table_of_contents from LLM was not a list, got {type(raw_toc).__name__}. Defaulting to empty list.")


                raw_sections = parsed_json.get("article_sections", [])
                final_sections = []
                if isinstance(raw_sections, list):
                    for section_data_from_llm in raw_sections:
                        if isinstance(section_data_from_llm, dict):
                            heading = str(section_data_from_llm.get("heading", "Missing Heading"))
                            # UPDATED to expect "content_block" and "timestamp_seconds"
                            content_block_text = str(section_data_from_llm.get("content_block", "Missing content"))
                            timestamp_val = section_data_from_llm.get("timestamp_seconds") # Can be float or null

                            # Validate or convert timestamp_val to float or None
                            final_timestamp_seconds = None
                            if isinstance(timestamp_val, (int, float)):
                                final_timestamp_seconds = float(timestamp_val)
                            elif timestamp_val is not None: # If it's not a number and not None, log a warning
                                print(f"Warning: Invalid timestamp_seconds value '{timestamp_val}' for heading '{heading}'. Setting to null.")


                            is_relevant_from_llm = section_data_from_llm.get("relevant_to_search_intent", False)
                            final_is_relevant = bool(is_relevant_from_llm) if user_search_intent else False

                            final_sections.append({
                                "heading": heading,
                                "content_block": content_block_text, # Use new key
                                "timestamp_seconds": final_timestamp_seconds, # Use new key
                                "relevant_to_search_intent": final_is_relevant
                            })
                        else:
                            print(f"Warning: An item in article_sections was not a dictionary, got {type(section_data_from_llm).__name__}. Skipping.")
                else:
                     print(f"Warning: article_sections from LLM was not a list, got {type(raw_sections).__name__}. Defaulting to empty list.")
                
                # Default error messages in case parsing leads to empty critical fields
                if not final_summary and not final_sections: # If both are empty, very likely a parsing or major LLM error
                     final_summary = "Error: Failed to extract meaningful content from LLM response."

                return LLMArticleOutput(
                    summary=final_summary,
                    table_of_contents=final_toc,
                    article_sections=final_sections
                )
            except json.JSONDecodeError as e:
                # ... (error handling remains similar, adjust field names in default error if necessary) ...
                error_msg = f"Error: Could not parse LLM JSON output. Details: {e}"
                print(f"LLM Service: {error_msg}")
                print(f"LLM Output that failed parsing (first 500 chars): {cleaned_llm_output_text[:500]}")
                return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])

            except Exception as e:
                # ... (error handling remains similar, adjust field names in default error if necessary) ...
                error_msg = f"Error: Unexpected issue processing LLM output. Details: {e} (Type: {type(e).__name__})"
                print(f"LLM Service: {error_msg}")
                return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])
        else:
            # ... (error handling remains similar, adjust field names in default error if necessary) ...
            error_msg = "Error: Unexpected LLM API response structure or empty content."
            print(f"LLM Service: {error_msg}")
            if response and hasattr(response, 'stop_reason'): print(f"API Response Stop Reason: {response.stop_reason}, Stop Sequence: {response.stop_sequence}")
            return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])


    except Exception as e: # Generic catch-all for API call level errors
        # ... (error handling remains similar, adjust field names in default error if necessary) ...
        error_msg = f"Error: An unexpected error occurred during LLM call. {e} (Type: {type(e).__name__})"
        print(f"LLM Service: {error_msg}")
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])