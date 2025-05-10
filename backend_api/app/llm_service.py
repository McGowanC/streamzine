import os
import anthropic
from pydantic import BaseModel
from typing import List, Dict, Any
import json

# Initialize the Anthropic client
# It will automatically look for the ANTHROPIC_API_KEY environment variable.
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


# Define the model name we'll use, making it configurable via environment variable
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "claude-3-haiku-20240307")

class LLMArticleOutput(BaseModel):
    summary: str
    table_of_contents: List[str] 
    article_sections: List[Dict[str, Any]] 

async def generate_article_from_transcript(
    transcript_segments: List[Dict[str, Any]], 
    user_search_intent: str | None, # This is the key input from the user
    video_title: str,
    model_name: str = DEFAULT_LLM_MODEL
) -> LLMArticleOutput:
    """
    Generates a structured article from video transcript segments using an LLM.
    """
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        print("LLM Service: Anthropic client not available or API key issue.")
        return LLMArticleOutput(
            summary="Error: LLM service not available or API key issue.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content": "LLM processing failed due to configuration.", "relevant_to_search_intent": False}]
        )

    if not transcript_segments:
        print("LLM Service: Transcript segments list is empty, cannot generate article.")
        return LLMArticleOutput(
            summary="Transcript was empty.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content": "Transcript was empty.", "relevant_to_search_intent": False}]
        )
    
    full_transcript_text = " ".join([segment.get('text', '') for segment in transcript_segments])
    if not full_transcript_text.strip():
        print("LLM Service: Combined transcript text is empty or only whitespace, cannot generate article.")
        return LLMArticleOutput(
            summary="Transcript text was effectively empty after combining segments.",
            table_of_contents=[],
            article_sections=[{"heading": "Error", "content": "Transcript text was empty.", "relevant_to_search_intent": False}]
        )

    system_prompt = """You are an expert summarizer and article writer. Your task is to transform a raw video transcript into a well-structured, engaging article. You have masterful prose that engages the reader.
The article should include:
1. A concise overall summary of the video content as a single string (e.g., a short paragraph or 3-5 key bullet points formatted as a single string with newlines).
2. A table of contents consisting of a list of strings, where each string is a main section heading.
3. The main article content, as a list of objects, where each object represents a section and has the following keys:
   - "heading": (string) The section heading.
   - "content": (string) The textual content of this section. Please try to embed original start timestamps like [HH:MM:SS] within this content where a new topic or significant point from the transcript begins. Convert seconds (e.g., 123.45s) to HH:MM:SS format for these embedded timestamps.
   - "relevant_to_search_intent": (boolean) True if this section is particularly relevant to the user's search intent (if one was provided), false otherwise. If no user_search_intent is provided, this should always be false.

Ensure the article flows well and reads like a coherent piece, not just a reformatted transcript. You don't have to incorporate every single piece of the transcript. Act like you are the creator of the youtube video. Don't say "then the speaker does", act is if the creator of the video is writing the article. Rephrase spoken language into more formal written language where appropriate.
If a user_search_intent is provided, pay close attention to it when determining relevance for 'relevant_to_search_intent'.

Output the result STRICTLY as a single JSON object with the exact keys: "summary", "table_of_contents", and "article_sections".
"""

    user_message_content = f"Video Title: \"{video_title}\"\n\n"
    user_message_content += "Transcript with timestamps (in seconds):\n"
    for segment in transcript_segments[:150]: 
        start_time = segment.get('start', 0.0)
        duration = segment.get('duration', 0.0)
        text = segment.get('text', '')
        user_message_content += f"- At {start_time:.2f}s (duration {duration:.2f}s): {text}\n"
    
    if len(transcript_segments) > 150:
        user_message_content += f"\n... (transcript truncated after {len(transcript_segments[:150])} segments from a total of {len(transcript_segments)} segments) ..."
    
    user_message_content += f"\n\nPlease transform this transcript into an article based on the system instructions."
    if user_search_intent: # Only add this part to the prompt if an intent was actually given
        user_message_content += f"\nThe user's search intent is: \"{user_search_intent}\". Please identify sections relevant to this intent and set 'relevant_to_search_intent' to true for those sections."
    else: # If no search intent, explicitly tell the LLM to set relevant_to_search_intent to false
        user_message_content += "\nNo specific user search intent was provided, so 'relevant_to_search_intent' should be false for all sections."

    
    user_message_content += "\n\nRemember to provide your output STRICTLY as a single JSON object with keys: 'summary' (string), 'table_of_contents' (list of strings), and 'article_sections' (list of objects, each with 'heading' (string), 'content' (string with embedded [HH:MM:SS] timestamps), and 'relevant_to_search_intent' (boolean))."

    print("\n--- Sending to LLM ---")
    
    try:
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
            print("\n--- LLM Raw Output (first 500 chars) ---")
            print(llm_output_text[:500])
            print("--- End LLM Raw Output ---")
            
            try:
                if llm_output_text.strip().startswith("```json"):
                    llm_output_text = llm_output_text.strip()[7:-3].strip()
                elif llm_output_text.strip().startswith("```"):
                     llm_output_text = llm_output_text.strip()[3:-3].strip()

                parsed_json = json.loads(llm_output_text)
                
                raw_summary = parsed_json.get("summary")
                final_summary = "Summary not provided or in unexpected format."
                if isinstance(raw_summary, str):
                    final_summary = raw_summary
                elif isinstance(raw_summary, list):
                    if len(raw_summary) > 0 and isinstance(raw_summary[0], str):
                        final_summary = raw_summary[0]
                    elif len(raw_summary) == 0:
                        final_summary = "Summary format error: Received an empty list."
                    else:
                        final_summary = f"Summary format error: Expected string in list, got {type(raw_summary[0]).__name__}."
                elif raw_summary is None:
                     final_summary = "Summary field was missing in LLM JSON output."
                
                raw_toc = parsed_json.get("table_of_contents")
                final_toc = []
                if isinstance(raw_toc, list):
                    final_toc = [str(item) for item in raw_toc if isinstance(item, str)]
                    if len(final_toc) != len(raw_toc):
                         print("Warning: Some items in table_of_contents were not strings and were excluded.")
                elif raw_toc is None:
                    print("Warning: table_of_contents field was missing in LLM JSON output.")
                else:
                    print(f"Warning: table_of_contents from LLM was not a list, got {type(raw_toc).__name__}. Defaulting to empty list.")

                raw_sections = parsed_json.get("article_sections")
                final_sections = []
                if isinstance(raw_sections, list):
                    for section_data_from_llm in raw_sections: # Renamed loop variable
                        if isinstance(section_data_from_llm, dict):
                            heading = section_data_from_llm.get("heading", "Missing Heading")
                            content = section_data_from_llm.get("content", "Missing Content")
                            
                            # --- MODIFICATION FOR RELEVANCE (Option 1B) ---
                            # Get the LLM's initial assessment
                            is_relevant_from_llm = section_data_from_llm.get("relevant_to_search_intent", False)
                            
                            # Override if no user_search_intent was provided to the function
                            if not user_search_intent: 
                                final_is_relevant = False 
                            else:
                                final_is_relevant = is_relevant_from_llm # Trust LLM if intent was given
                            # --- END MODIFICATION ---
                            
                            if not isinstance(heading, str): heading = str(heading)
                            if not isinstance(content, str): content = str(content)
                            if not isinstance(final_is_relevant, bool): final_is_relevant = False # Default if not boolean

                            final_sections.append({
                                "heading": heading,
                                "content": content,
                                "relevant_to_search_intent": final_is_relevant 
                            })
                        else:
                            print(f"Warning: An item in article_sections was not a dictionary, got {type(section_data_from_llm).__name__}. Skipping.")
                elif raw_sections is None:
                    print("Warning: article_sections field was missing in LLM JSON output.")
                else:
                    print(f"Warning: article_sections from LLM was not a list, got {type(raw_sections).__name__}. Defaulting to empty list.")
                
                return LLMArticleOutput(
                    summary=final_summary,
                    table_of_contents=final_toc,
                    article_sections=final_sections
                )
            except json.JSONDecodeError as e:
                error_msg = f"Error: Could not parse LLM JSON output. Details: {e}"
                print(f"LLM Service: {error_msg}")
                print(f"LLM Output that failed parsing (first 500 chars): {llm_output_text[:500]}")
                return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])
            except Exception as e: 
                error_msg = f"Error: Unexpected issue processing LLM output. Details: {e} (Type: {type(e).__name__})"
                print(f"LLM Service: {error_msg}")
                return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])
        else:
            error_msg = "Error: Unexpected LLM API response structure or empty content."
            print(f"LLM Service: {error_msg}")
            if response: print(f"API Response Stop Reason: {response.stop_reason}, Stop Sequence: {response.stop_sequence}")
            return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])

    except anthropic.APIConnectionError as e:
        error_msg = f"Error: Anthropic API connection error. {e}"
        print(f"LLM Service: {error_msg}")
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])
    except anthropic.RateLimitError as e:
        error_msg = f"Error: Anthropic API rate limit exceeded. {e}"
        print(f"LLM Service: {error_msg}")
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])
    except anthropic.APIStatusError as e:
        error_msg = f"Error: Anthropic API returned an error status {e.status_code}. Message: {e.message}"
        print(f"LLM Service: {error_msg}")
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])
    except Exception as e: 
        error_msg = f"Error: An unexpected error occurred during LLM call. {e} (Type: {type(e).__name__})"
        print(f"LLM Service: {error_msg}")
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[])