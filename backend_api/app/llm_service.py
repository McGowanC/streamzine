import os

import logging
import httpx # Make sure httpx is imported if not already

# Configure basic logging to see DEBUG messages from httpx
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG) # Also log httpcore

import anthropic
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple 
import json
import re # For simple token estimation, or we can use tiktoken later

# Initialize the Anthropic client
try:
    client = anthropic.AsyncAnthropic() # Use AsyncAnthropic
    ANTHROPIC_API_KEY_AVAILABLE = True
    print("LLM Service: Async Anthropic client initialized successfully.")
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
SEMANTIC_BOUNDARY_MODEL = os.getenv("SEMANTIC_BOUNDARY_MODEL", DEFAULT_LLM_MODEL)
SECTION_GENERATION_MODEL = os.getenv("SECTION_GENERATION_MODEL", DEFAULT_LLM_MODEL)
FINAL_SUMMARY_MODEL = os.getenv("FINAL_SUMMARY_MODEL", DEFAULT_LLM_MODEL)

# --- NEW CONSTANTS for Semantic Chunking & Batching ---
MIN_SEGMENTS_FOR_SEMANTIC_CHUNKING = 50 
MAX_INPUT_TOKENS_STAGE0_BOUNDARY_ID = 180000 
MAX_TRANSCRIPT_TOKENS_PER_STAGE1_BATCH = 10000 # Adjusted placeholder, might need further tuning (was 15000)
MAX_SEMANTIC_CHUNKS_PER_STAGE1_BATCH = 5    # Adjusted placeholder (was 7)
MAX_RETRIES_STAGE1_TRUNCATION = 2           

# --- Pydantic Models (existing) ---
class LLMArticleOutput(BaseModel):
    summary: str
    table_of_contents: List[str]
    article_sections: List[Dict[str, Any]]

class SemanticBoundary(BaseModel):
    start_segment_index: int
    end_segment_index: int
    brief_topic_hint: str | None = None

# --- Helper for Token Estimation (placeholder) ---
def estimate_tokens(text: str) -> int:
    return len(text) // 3 

# --- Stage 0: Semantic Boundary Identification ---
async def identify_semantic_boundaries(
    transcript_segments: List[Dict[str, Any]],
    video_title: str,
    model_name: str = SEMANTIC_BOUNDARY_MODEL
) -> List[SemanticBoundary] | None:
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        print("LLM Service (Stage 0): Client not available.")
        return None
    if not transcript_segments:
        print("LLM Service (Stage 0): No transcript segments to process.")
        return None

    print(f"LLM Service (Stage 0): Identifying semantic boundaries for '{video_title}'.")
    transcript_for_llm = ""
    for i, segment in enumerate(transcript_segments):
        transcript_for_llm += f"[Segment {i} | Time {segment.get('start', 0.0):.2f}s]: {segment.get('text', '')}\n"

    estimated_input_tokens = estimate_tokens(transcript_for_llm)
    if estimated_input_tokens > MAX_INPUT_TOKENS_STAGE0_BOUNDARY_ID:
        print(f"LLM Service (Stage 0): Transcript too long for semantic boundary identification ({estimated_input_tokens} estimated tokens > {MAX_INPUT_TOKENS_STAGE0_BOUNDARY_ID}). Skipping this stage.")
        return [SemanticBoundary(start_segment_index=0, end_segment_index=len(transcript_segments)-1, brief_topic_hint="Full video content")]

    system_prompt = """Your task is to analyze the provided YouTube video transcript and identify logical semantic boundaries or topic shifts.
The transcript is formatted with `[Segment INDEX | Time TIMESTAMPs]: TEXT`.
You should output a JSON list of objects. Each object must represent a distinct semantic section and have the following keys:
- "start_segment_index": (integer) The index of the first segment in this semantic section.
- "end_segment_index": (integer) The index of the last segment in this semantic section.
- "brief_topic_hint": (string, optional) A very short (2-5 words) hint about the main topic of this section.
Ensure the sections are contiguous and cover the entire transcript. The `end_segment_index` of one section should typically be followed by a `start_segment_index` of the next that is `end_segment_index + 1`.
Do not overlap sections. Strive for meaningful sections, not too short and not too long. A typical section might span 10-50 transcript segments, but this can vary greatly based on content.
The goal is to divide the transcript into coherent parts that can later be processed individually to build an article.
Output ONLY the JSON list of objects.
Example:
[
  {"start_segment_index": 0, "end_segment_index": 15, "brief_topic_hint": "Introduction to topic X"},
  {"start_segment_index": 16, "end_segment_index": 45, "brief_topic_hint": "Core concepts of X"},
  {"start_segment_index": 46, "end_segment_index": 59, "brief_topic_hint": "Implications of X"}
]
"""
    user_message_content = f"Video Title: \"{video_title}\"\n\nTranscript:\n{transcript_for_llm}\n\nPlease identify semantic boundaries and output the JSON list."

    print(f"LLM Service (Stage 0): Sending {len(transcript_segments)} segments to LLM for boundary identification.")
    try:
        response = await client.messages.create(
            model=model_name,
            max_tokens=2048, 
            system=system_prompt,
            messages=[{"role": "user", "content": user_message_content}]
        )

        if response.stop_reason == "max_tokens":
            print("LLM Service (Stage 0): Truncation detected (max_tokens). Failed to get complete boundaries.")
            return None 

        if response.content and response.content[0].text:
            llm_output_text = response.content[0].text.strip()
            if llm_output_text.startswith("```json"):
                llm_output_text = llm_output_text[7:]
                if llm_output_text.endswith("```"):
                    llm_output_text = llm_output_text[:-3]
            elif llm_output_text.startswith("```"):
                llm_output_text = llm_output_text[3:]
                if llm_output_text.endswith("```"):
                    llm_output_text = llm_output_text[:-3]
            llm_output_text = llm_output_text.strip()

            try:
                parsed_boundaries_raw = json.loads(llm_output_text)
                boundaries = []
                if not isinstance(parsed_boundaries_raw, list):
                    print(f"LLM Service (Stage 0): Expected a list of boundaries, got {type(parsed_boundaries_raw)}. Treating as failure.")
                    return None

                for item in parsed_boundaries_raw:
                    if isinstance(item, dict) and \
                       isinstance(item.get("start_segment_index"), int) and \
                       isinstance(item.get("end_segment_index"), int):
                        boundaries.append(SemanticBoundary(
                            start_segment_index=item["start_segment_index"],
                            end_segment_index=item["end_segment_index"],
                            brief_topic_hint=item.get("brief_topic_hint")
                        ))
                    else:
                        print(f"LLM Service (Stage 0): Invalid boundary item format: {item}. Skipping.")
                
                if not boundaries: 
                    print("LLM Service (Stage 0): No valid boundaries parsed from LLM response.")
                    return None
                
                print(f"LLM Service (Stage 0): Successfully identified {len(boundaries)} semantic boundaries.")
                return boundaries
            except json.JSONDecodeError as e:
                print(f"LLM Service (Stage 0): JSONDecodeError parsing boundaries. Stop Reason: {response.stop_reason}. Error: {e}")
                print(f"LLM Output (Stage 0) that failed parsing: {llm_output_text[:1000]}")
                return None
        else:
            print(f"LLM Service (Stage 0): Empty or unexpected response content. Stop Reason: {response.stop_reason}")
            return None

    except anthropic.APIError as e:
        print(f"LLM Service (Stage 0): Anthropic API error: {e}")
        return None
    except Exception as e:
        print(f"LLM Service (Stage 0): Unexpected error: {type(e).__name__} - {e}")
        return None

# --- Stage 1: Article Section Generation from Batched Semantic Chunks ---
async def generate_sections_from_batched_chunks(
    transcript_segments: List[Dict[str, Any]],
    semantic_boundaries: List[SemanticBoundary],
    video_title: str,
    user_search_intent: str | None, # Explicitly check for None or empty string
    model_name: str = SECTION_GENERATION_MODEL
) -> List[Dict[str, Any]]: 
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        print("LLM Service (Stage 1): Client not available.")
        return [{"heading": "Error", "content_block": "LLM Client not available for section generation.", "timestamp_seconds": None, "relevant_to_search_intent": False}]
    if not transcript_segments or not semantic_boundaries:
        print("LLM Service (Stage 1): No transcript or boundaries to process.")
        return []

    all_generated_article_sections: List[Dict[str, Any]] = []
    current_batch_semantic_chunks: List[SemanticBoundary] = []
    current_batch_transcript_text = ""
    current_batch_estimated_tokens = 0
    boundary_idx = 0

    # Normalize user_search_intent: treat empty string as None for logic checks
    effective_search_intent = user_search_intent if user_search_intent and user_search_intent.strip() else None


    while boundary_idx < len(semantic_boundaries):
        semantic_chunk_to_add = semantic_boundaries[boundary_idx]
        chunk_text = ""
        start_idx = max(0, semantic_chunk_to_add.start_segment_index)
        end_idx = min(len(transcript_segments) -1 , semantic_chunk_to_add.end_segment_index)

        if start_idx > end_idx:
             print(f"LLM Service (Stage 1): Invalid boundary segment indices {start_idx}-{end_idx}. Skipping boundary.")
             boundary_idx += 1
             continue

        for i in range(start_idx, end_idx + 1):
            chunk_text += transcript_segments[i].get('text', '') + "\n"
        
        estimated_tokens_for_chunk_to_add = estimate_tokens(chunk_text)
        can_add_to_batch = True
        if not current_batch_semantic_chunks: 
            can_add_to_batch = True
        else:
            if (current_batch_estimated_tokens + estimated_tokens_for_chunk_to_add) > MAX_TRANSCRIPT_TOKENS_PER_STAGE1_BATCH:
                can_add_to_batch = False
                print(f"LLM Service (Stage 1): Batch token limit {MAX_TRANSCRIPT_TOKENS_PER_STAGE1_BATCH} would be exceeded by adding next chunk. Processing current batch first.")
            if len(current_batch_semantic_chunks) >= MAX_SEMANTIC_CHUNKS_PER_STAGE1_BATCH:
                can_add_to_batch = False
                print(f"LLM Service (Stage 1): Max semantic chunks per batch {MAX_SEMANTIC_CHUNKS_PER_STAGE1_BATCH} reached. Processing current batch first.")

        if can_add_to_batch:
            current_batch_semantic_chunks.append(semantic_chunk_to_add)
            current_batch_transcript_text += chunk_text
            current_batch_estimated_tokens += estimated_tokens_for_chunk_to_add
            boundary_idx += 1
        
        if not current_batch_semantic_chunks and not can_add_to_batch and boundary_idx < len(semantic_boundaries):
            # This case means a single semantic chunk itself is too large based on estimated_tokens_for_chunk_to_add
            # or other criteria that prevented it from forming even a batch of one.
            # We'll try to process it alone; it might still fail if it's truly massive for output.
            print(f"LLM Service (Stage 1): Single semantic chunk from S.{semantic_chunk_to_add.start_segment_index}-S.{semantic_chunk_to_add.end_segment_index} (est. {estimated_tokens_for_chunk_to_add} tokens) is being processed alone as it exceeds batch formation criteria or is last.")
            current_batch_semantic_chunks.append(semantic_chunk_to_add)
            current_batch_transcript_text = chunk_text 
            current_batch_estimated_tokens = estimated_tokens_for_chunk_to_add
            can_add_to_batch = False # Force processing of this single large chunk
            boundary_idx +=1 # Ensure we move to the next boundary after this attempt


        if (not can_add_to_batch or boundary_idx == len(semantic_boundaries)) and current_batch_semantic_chunks:
            print(f"LLM Service (Stage 1): Processing batch of {len(current_batch_semantic_chunks)} semantic chunks. Estimated transcript tokens for batch: {current_batch_estimated_tokens}.")

            batch_prompt_intro = f"Video Title: \"{video_title}\"\n"
            if effective_search_intent: # Use normalized version
                batch_prompt_intro += f"User Search Intent: \"{effective_search_intent}\". Highlight relevant sections by setting 'relevant_to_search_intent' to true.\n"
            else:
                # ***** MODIFICATION: Stronger instruction when no search intent *****
                batch_prompt_intro += "No specific user search intent was provided. Therefore, the 'relevant_to_search_intent' field in ALL generated article_section objects MUST be set to false.\n"

            batch_prompt_intro += "\nThe following transcript content has been pre-analyzed into semantic topics/sections. For EACH pre-identified topic below, generate one or more logical 'article_sections'.\n"
            topic_hints_for_prompt = ""
            for i, sc_boundary in enumerate(current_batch_semantic_chunks):
                start_time_seg_idx = max(0, min(sc_boundary.start_segment_index, len(transcript_segments)-1))
                end_time_seg_idx = max(0, min(sc_boundary.end_segment_index, len(transcript_segments)-1))
                
                start_time = transcript_segments[start_time_seg_idx].get('start', 0.0)
                end_seg_start = transcript_segments[end_time_seg_idx].get('start', 0.0)
                end_seg_duration = transcript_segments[end_time_seg_idx].get('duration', 1.0) 
                end_time = end_seg_start + end_seg_duration
                topic_hints_for_prompt += f"- Topic Hint {i+1}: \"{sc_boundary.brief_topic_hint or 'N/A'}\" (Original segments S.{sc_boundary.start_segment_index}-S.{sc_boundary.end_segment_index}; Approx Time {start_time:.2f}s to {end_time:.2f}s)\n"
            
            system_prompt_stage1 = f"""{batch_prompt_intro}
{topic_hints_for_prompt}
Each 'article_section' object you generate must have the following JSON keys:
- "heading": (string) A descriptive heading for this specific content block or paragraph. This should reflect the topic hint if relevant.
- "content_block": (string) The textual content for this block. THIS TEXT SHOULD NOT CONTAIN ANY EMBEDDED TIMESTAMPS.
- "timestamp_seconds": (float or null) The single most relevant start time in TOTAL SECONDS (e.g., 123.45) from the original transcript that corresponds to the beginning of this 'content_block'. This must be derived from the original segment times.
- "relevant_to_search_intent": (boolean) True if this content_block is particularly relevant to the user's search intent (MUST be false if no intent was provided as stated above).
The 'content_block' should be well-written and flow naturally. Rephrase spoken language into formal written language where appropriate.
Organize the output based on the provided topic hints.
Output STRICTLY a single JSON list of these 'article_section' objects for all provided topics in this batch. Do not include any other text before or after the JSON list.
Example of a single item in the list:
  {{"heading": "Example Heading", "content_block": "This is the processed text for this section.", "timestamp_seconds": 45.88, "relevant_to_search_intent": false}}
"""
            user_message_content_stage1 = f"Transcript segments for this batch:\n{current_batch_transcript_text}\n\nPlease generate the JSON list of 'article_section' objects as instructed in the system prompt."

            retries = 0
            batch_processed_successfully = False
            # TODO: Implement more robust retry that splits current_batch_semantic_chunks if it fails
            # For now, the retry logic is basic and re-attempts the same (potentially problematic) batch.
            # The log message "Retry logic needs to correctly reconstruct input..." highlights this gap.
            
            # For the retry logic, we will focus on the current_batch_semantic_chunks
            # If it fails, the more robust retry would involve breaking *this* list down further
            # and calling this inner processing part recursively or iteratively for smaller pieces.
            # This is a placeholder for that more complex retry.
            # The current retry just re-attempts the whole batch.

            temp_batch_to_process = list(current_batch_semantic_chunks) # This is what we'd split in a more advanced retry

            while retries <= MAX_RETRIES_STAGE1_TRUNCATION and not batch_processed_successfully:
                # If retries > 0, this is where more advanced logic would split temp_batch_to_process
                # and reconstruct system_prompt_stage1 and user_message_content_stage1
                # For now, we're retrying the same content.
                if retries > 0:
                    print(f"LLM Service (Stage 1): Retrying batch processing (attempt {retries}). Current retry is basic and re-attempts same batch content.")
                    # To make retry slightly more useful, if it's a multi-chunk batch that failed,
                    # a very simple "split" is to try only the first chunk of that batch.
                    if len(temp_batch_to_process) > 1: # Only if it was a multi-chunk batch
                        print(f"LLM Service (Stage 1): Simplistic retry: attempting only first semantic chunk of the failed batch.")
                        # This is a HACK and not a full robust retry for now.
                        # A proper retry needs to re-evaluate token counts, reconstruct prompts for the sub-chunk.
                        # For the purpose of this focused fix, we'll let it retry with original prompt but signal it.
                        # This part will still likely fail if the original issue was output size for original input size.
                    elif not temp_batch_to_process:
                        print("LLM Service (Stage 1): No chunks to retry. Aborting for this batch.")
                        break


                try:
                    print(f"LLM Service (Stage 1): Sending batch (SC count: {len(temp_batch_to_process)}) to LLM. Retry: {retries}")
                    response = await client.messages.create(
                        model=model_name,
                        max_tokens=4000, 
                        system=system_prompt_stage1, # This prompt is for the whole temp_batch_to_process
                        messages=[{"role": "user", "content": user_message_content_stage1}] # Content is also for whole temp_batch
                    )

                    if response.stop_reason == "max_tokens":
                        print(f"LLM Service (Stage 1): Truncation detected (max_tokens) for batch. Retry {retries}/{MAX_RETRIES_STAGE1_TRUNCATION}.")
                        # (Error section added outside loop if all retries fail)
                    elif response.content and response.content[0].text:
                        llm_output_text = response.content[0].text.strip()
                        if llm_output_text.startswith("```json"):
                            llm_output_text = llm_output_text[7:]
                            if llm_output_text.endswith("```"):
                                llm_output_text = llm_output_text[:-3]
                        elif llm_output_text.startswith("```"):
                            llm_output_text = llm_output_text[3:]
                            if llm_output_text.endswith("```"):
                                llm_output_text = llm_output_text[:-3]
                        llm_output_text = llm_output_text.strip()
                        
                        try:
                            parsed_sections_from_llm = json.loads(llm_output_text)
                            if isinstance(parsed_sections_from_llm, list):
                                processed_sections_for_batch = []
                                for section_data in parsed_sections_from_llm:
                                    if isinstance(section_data, dict):
                                        # ***** MODIFICATION: Post-processing override for relevant_to_search_intent *****
                                        if not effective_search_intent: # Check original user_search_intent
                                            section_data["relevant_to_search_intent"] = False
                                        # Basic validation of expected keys
                                        if not all(k in section_data for k in ["heading", "content_block", "timestamp_seconds", "relevant_to_search_intent"]):
                                            print(f"LLM Service (Stage 1): Warning - LLM output section missing keys: {section_data}")
                                            # Add with defaults or skip? For now, add as is, frontend should handle missing keys gracefully if possible.
                                        processed_sections_for_batch.append(section_data)
                                    else:
                                        print(f"LLM Service (Stage 1): Warning - item in parsed_sections_from_llm is not a dict: {section_data}")
                                
                                all_generated_article_sections.extend(processed_sections_for_batch)
                                batch_processed_successfully = True
                                print(f"LLM Service (Stage 1): Successfully processed batch, got {len(processed_sections_for_batch)} article sections.")
                            else:
                                print(f"LLM Service (Stage 1): Expected a list of sections, got {type(parsed_sections_from_llm)}. Stop Reason: {response.stop_reason}. Retry {retries}/{MAX_RETRIES_STAGE1_TRUNCATION}")
                        except json.JSONDecodeError as e:
                            print(f"LLM Service (Stage 1): JSONDecodeError parsing sections for batch. Stop Reason: {response.stop_reason}. Error: {e}. Retry {retries}/{MAX_RETRIES_STAGE1_TRUNCATION}")
                            print(f"LLM Output (Stage 1) that failed: {llm_output_text[:500]}")
                    else: 
                        print(f"LLM Service (Stage 1): Empty or unexpected LLM response content for batch. Stop Reason: {response.stop_reason}. Retry {retries}/{MAX_RETRIES_STAGE1_TRUNCATION}")
                    
                    if not batch_processed_successfully:
                        retries += 1

                except anthropic.APIError as e:
                    print(f"LLM Service (Stage 1): Anthropic API error processing batch: {e}")
                    retries += 1 
                except Exception as e:
                    print(f"LLM Service (Stage 1): Unexpected error processing batch: {type(e).__name__} - {e}")
                    retries +=1
            
            if not batch_processed_successfully:
                # Add error section for the temp_batch_to_process that failed all retries
                failed_chunk_ids_str = ", ".join([f"S.{s.start_segment_index}-S.{s.end_segment_index}" for s in temp_batch_to_process])
                error_content = f"Content for semantic chunks ({failed_chunk_ids_str}) could not be processed after multiple retries."
                if temp_batch_to_process : # check if list is not empty
                    error_ts = transcript_segments[max(0,temp_batch_to_process[0].start_segment_index)].get('start')
                else: # should not happen if current_batch_semantic_chunks had items
                    error_ts = None
                all_generated_article_sections.append({
                    "heading": "Batch Processing Error",
                    "content_block": error_content,
                    "timestamp_seconds": error_ts,
                    "relevant_to_search_intent": False
                })
            
            current_batch_semantic_chunks = []
            current_batch_transcript_text = ""
            current_batch_estimated_tokens = 0
            
    return all_generated_article_sections

# --- Stage 2: Final Aggregation - Summary and Table of Contents ---
async def generate_final_summary_and_toc(
    all_article_sections: List[Dict[str, Any]],
    video_title: str,
    model_name: str = FINAL_SUMMARY_MODEL
) -> Tuple[str, List[str]]: 
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        print("LLM Service (Stage 2): Client not available.")
        return "Error: LLM client not available for final summary.", []
    
    # Filter out any placeholder error sections before sending to Stage 2,
    # unless it's the *only* thing we have.
    valid_sections_for_summary = [
        s for s in all_article_sections 
        if isinstance(s, dict) and not (
            s.get("heading", "").lower().startswith("batch processing error") or 
            s.get("heading", "").lower().startswith("api error during") or
            s.get("heading", "").lower().startswith("unexpected error during")
        )
    ]
    if not valid_sections_for_summary and all_article_sections: # If only error sections exist, use them
        print("LLM Service (Stage 2): Only error sections found. Summarizing error state.")
        content_for_summary_prompt = f"Video Title: \"{video_title}\"\n\nThe following errors occurred during processing:\n"
        for i, section in enumerate(all_article_sections): # Use all_article_sections here
            heading = section.get("heading", f"Error {i+1}")
            content_snippet = section.get("content_block", "No details.")[:250] + "..."
            content_for_summary_prompt += f"- Error: \"{heading}\"\n  Detail: \"{content_snippet}\"\n"
    elif not all_article_sections:
        print("LLM Service (Stage 2): No article sections (valid or error) to summarize.")
        return "No content was processed to generate a summary.", []
    else: # We have valid sections
        print(f"LLM Service (Stage 2): Generating final summary and ToC for '{video_title}' from {len(valid_sections_for_summary)} valid sections.")
        content_for_summary_prompt = f"Video Title: \"{video_title}\"\n\nThe following article sections (headings and brief content snippets) were generated from the video:\n"
        temp_token_count = estimate_tokens(content_for_summary_prompt)
        max_tokens_for_stage2_input_sections = 30000 
        section_details_for_prompt = ""
        sections_sent_count = 0
        for section in valid_sections_for_summary: # Use valid_sections_for_summary
            heading = section.get("heading", "Untitled Section")
            content_snippet = (section.get("content_block", "")[:150] + "...") if section.get("content_block") else "N/A"
            detail = f"- Heading: \"{heading}\"\n  Snippet: \"{content_snippet}\"\n"
            estimated_detail_tokens = estimate_tokens(detail)
            if temp_token_count + estimated_detail_tokens > max_tokens_for_stage2_input_sections:
                print(f"LLM Service (Stage 2): Reached token limit for section details. Sending {sections_sent_count} section details for summary.")
                break
            section_details_for_prompt += detail
            temp_token_count += estimated_detail_tokens
            sections_sent_count += 1
        if not section_details_for_prompt and valid_sections_for_summary: 
            section_details_for_prompt = "Many sections were generated, but details are omitted for brevity in this summary request.\n"
            for i, section in enumerate(valid_sections_for_summary[:10]): 
                 section_details_for_prompt += f"- Heading: \"{section.get('heading', f'Section {i+1}')}\"\n"
        content_for_summary_prompt += section_details_for_prompt

    system_prompt_stage2 = """You are an expert summarizer. Based on the provided video title and a list of article section headings (and possibly content snippets) from that video, your tasks are:
1.  Generate a concise overall summary of the video's main points as a single string. If the input mainly consists of error messages, summarize the error state.
2.  Create a table of contents (ToC) as a list of strings. Each string in the list should be a main section heading suitable for a ToC. These should be derived from the provided section headings. If the input is primarily errors, the ToC might reflect that or be empty.
Output ONLY a single JSON object with the exact keys: "summary" (string) and "table_of_contents" (list of strings).
Example (successful):
{
  "summary": "This video discusses the key aspects of X, Y, and Z, highlighting their importance.",
  "table_of_contents": ["Introduction to X", "Exploring Y", "The Impact of Z", "Conclusion"]
}
Example (error state):
{
  "summary": "Processing of the video encountered errors, particularly during batch section generation.",
  "table_of_contents": ["Processing Errors"]
}
"""
    user_message_content_stage2 = f"{content_for_summary_prompt}\n\nPlease generate the summary and table of contents in the specified JSON format."

    try:
        response = await client.messages.create(
            model=model_name,
            max_tokens=1024, 
            system=system_prompt_stage2,
            messages=[{"role": "user", "content": user_message_content_stage2}]
        )

        if response.stop_reason == "max_tokens":
            print("LLM Service (Stage 2): Truncation detected (max_tokens) for summary/ToC.")
            return "Error: Summary/ToC generation was truncated by max_tokens.", []

        if response.content and response.content[0].text:
            llm_output_text = response.content[0].text.strip()
            if llm_output_text.startswith("```json"):
                llm_output_text = llm_output_text[7:]
                if llm_output_text.endswith("```"):
                    llm_output_text = llm_output_text[:-3]
            elif llm_output_text.startswith("```"):
                llm_output_text = llm_output_text[3:]
                if llm_output_text.endswith("```"):
                    llm_output_text = llm_output_text[:-3]
            llm_output_text = llm_output_text.strip()

            try:
                parsed_output = json.loads(llm_output_text)
                final_summary = parsed_output.get("summary", "Summary not provided by LLM.")
                final_toc = parsed_output.get("table_of_contents", [])

                if not isinstance(final_summary, str): final_summary = str(final_summary) 
                if not isinstance(final_toc, list): final_toc = [] 
                else: final_toc = [str(item) for item in final_toc if isinstance(item, str)] 

                print("LLM Service (Stage 2): Successfully generated summary and ToC.")
                return final_summary, final_toc
            except json.JSONDecodeError as e:
                print(f"LLM Service (Stage 2): JSONDecodeError parsing summary/ToC. Stop Reason: {response.stop_reason}. Error: {e}")
                print(f"LLM Output (Stage 2) that failed parsing: {llm_output_text[:500]}")
                return f"Error: Could not parse summary/ToC from LLM. Raw text received: {llm_output_text[:100]}...", []
        else:
            print(f"LLM Service (Stage 2): Empty or unexpected LLM response for summary/ToC. Stop Reason: {response.stop_reason}")
            return "Error: Empty response from LLM for summary/ToC.", []

    except anthropic.APIError as e:
        print(f"LLM Service (Stage 2): Anthropic API error: {e}")
        return f"Error: API error during summary/ToC generation - {str(e)[:100]}", []
    except Exception as e:
        print(f"LLM Service (Stage 2): Unexpected error: {type(e).__name__} - {e}")
        return f"Error: Unexpected error during summary/ToC generation - {str(e)[:100]}", []


# --- Main Orchestrator Function ---
async def generate_article_from_transcript( 
    transcript_segments: List[Dict[str, Any]],
    user_search_intent: str | None,
    video_title: str,
    model_name: str = DEFAULT_LLM_MODEL 
) -> LLMArticleOutput:
    if not ANTHROPIC_API_KEY_AVAILABLE or client is None:
        error_msg = "Error: LLM service not available or API key issue."
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])
    if not transcript_segments:
        error_msg = "Transcript was empty."
        return LLMArticleOutput(summary=error_msg, table_of_contents=[], article_sections=[{"heading":"Error", "content_block":error_msg, "timestamp_seconds":None, "relevant_to_search_intent":False }])

    print(f"\n--- Starting Multi-Stage Article Generation for: {video_title} ---")
    semantic_boundaries = None
    if len(transcript_segments) >= MIN_SEGMENTS_FOR_SEMANTIC_CHUNKING:
        semantic_boundaries = await identify_semantic_boundaries(
            transcript_segments, video_title
        )
    
    if not semantic_boundaries:
        print("LLM Service (Orchestrator): Failed to get semantic boundaries or transcript too short. Processing as a single block.")
        semantic_boundaries = [
            SemanticBoundary(start_segment_index=0, end_segment_index=len(transcript_segments) - 1, brief_topic_hint="Full video content")
        ]

    all_article_sections = await generate_sections_from_batched_chunks(
        transcript_segments,
        semantic_boundaries,
        video_title,
        user_search_intent 
    )

    final_summary, final_toc = await generate_final_summary_and_toc(
        all_article_sections, # Stage 2 will now filter errors internally if needed
        video_title
    )
    
    # If all_article_sections only contains errors, the summary might reflect that.
    # The goal is to always return a valid LLMArticleOutput structure.
    # If all_article_sections is empty (e.g. Stage 1 returned empty list on critical failure), 
    # summary might be like "No content processed..."
    # This logic can be refined based on how we want to present total failures vs partial.

    print(f"--- Multi-Stage Article Generation Completed for: {video_title} ---")
    return LLMArticleOutput(
        summary=final_summary,
        table_of_contents=final_toc,
        article_sections=all_article_sections # This might contain error sections if Stage 1 had partial failures
    )