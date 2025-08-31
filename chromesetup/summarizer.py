import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import openai
import logging
import asyncpraw
import asyncprawcore
import ssl
import certifi
import aiohttp
import time
from openai import OpenAIError
from pydantic import BaseModel
from typing import List
import asciidoc

# Additional libraries for HTML parsing, summarization, OCR, YouTube transcripts
import requests
import re
from bs4 import BeautifulSoup
from readability import Document  # pip install readability-lxml
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# OCR + Image
import pytesseract
from PIL import Image
from io import BytesIO

# YouTube Transcript
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# OpenCV for chunk-based video frame extraction
import cv2
import base64

# Add to imports section
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import tempfile

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS – you can add multiple allowed origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://dhgjompomnkacadcomfjemhancdmdkmd",  # Your extension
        "chrome-extension://*",  # Possibly any extension ID
        "http://localhost:8100",  # If you happen to test from another local port
        "http://localhost:3000"   # If you have a frontend dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Specifically silence SSL-related debug messages
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# Initialize OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables")
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

openai.api_key = openai_api_key

# Initialize ChatOpenAI model
# Replace "gpt-4o-mini-2024-07-18" with your actual model name, e.g. "gpt-4"
llm = ChatOpenAI(
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.5,
    openai_api_key=openai_api_key
)

####################################################
# --------------- Chat Endpoint ------------------- #
####################################################

# We'll store the entire conversation in memory for demonstration
conversation_history: List[dict] = []
post_context = ""  # Store the current post context

class ChatRequest(BaseModel):
    user_message: str
    post_content: str = ""  # Make it optional with default empty string

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    """
    A chat endpoint that processes user messages and returns bot replies.
    """
    global post_context, conversation_history
    user_input = chat_req.user_message
    logger.info(f"Received user message: {user_input}")

    try:
        # Update post context if new content is received
        if chat_req.post_content:
            # Clear previous conversation when post changes
            if chat_req.post_content != post_context:
                logger.info("New post content received, resetting conversation")
                conversation_history.clear()
                post_context = chat_req.post_content
                
                # Add system message with post context
                conversation_history.append({
                    "role": "system",
                    "content": f"""You are a concise assistant discussing a Reddit post. Here's the post content:

{post_context}

Guidelines for your responses:
- Keep responses short and focused (2-4 sentences maximum)
- Be direct and get straight to the point
- Avoid unnecessary details or elaboration
- Use simple, clear language
- Only provide longer responses if explicitly requested

Please help the user understand and discuss this post while following these guidelines."""
                })
                logger.info("Added system message with post context")

        # If no post context is set, use a default system message
        if not conversation_history:
            conversation_history.append({
                "role": "system",
                "content": "You are a concise assistant. Keep responses short and focused."
            })

        # Add the user's message to the conversation
        conversation_history.append({"role": "user", "content": user_input})

        # Format messages for ChatOpenAI
        messages = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in conversation_history
        ]

        # Use invoke method instead of direct call
        response = llm.invoke(messages)
        bot_reply = response.content
        logger.info(f"Generated bot reply: {bot_reply}")

        # Add the assistant's reply to conversation memory
        conversation_history.append({"role": "assistant", "content": bot_reply})

        return {"bot_reply": bot_reply}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"bot_reply": f"Sorry, I encountered an error: {str(e)}"}

####################################################
# ----------- Summarization Endpoints ------------- #
####################################################

# Define the prompt templates
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Please provide a summary of the following text. 

If this is a linked article, start your response with "The link shows" and then continue with a brief, focused summary in 2-3 sentences.

If this is the main post, provide a summary proportional to the text length:
- For short posts (< 500 words): 2-3 sentences
- For medium posts (500-1500 words): 4-6 sentences
- For long posts (> 1500 words): 6-8 sentences

Keep the summary informative and well-structured. Maintain all key points and important details.

Text to summarize:
{text}"""
)

comment_analysis_prompt = PromptTemplate(
    input_variables=["comments"],
    template="""Analyze these Reddit comments and provide a clear analysis. Your response length should be proportional to the amount of comment content:
    - For few comments (< 10): 2-3 sentences
    - For moderate comments (10-30): 4-5 sentences
    - For many comments (> 30): 6-8 sentences
    
    Focus on:
    - The main consensus/prevalent opinions
    - Any notable outlier perspectives or unique insights

    Comments to analyze:
    {comments}"""
)

os.environ['SSL_CERT_FILE'] = certifi.where()  # Use a valid certificate store

# Initialize Reddit API client
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_API_KEY"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

###################
# HELPER FUNCTIONS
###################

def get_word_count(text: str) -> int:
    return len(text.split())

def get_summary_length_guidance(word_count: int) -> str:
    if word_count < 500:
        return "2-3 concise sentences"
    elif word_count < 1500:
        return "4-6 detailed sentences"
    elif word_count < 3000:
        return "7-10 comprehensive sentences"
    else:
        return "10-15 detailed sentences, organized into key points"

def get_comment_length_guidance(comment_count: int, total_words: int) -> str:
    if comment_count < 10 or total_words < 500:
        return "2-3 sentences focusing on the main points"
    elif comment_count < 30 or total_words < 1500:
        return "5-7 sentences covering main consensus and notable opinions"
    else:
        return "8-12 sentences, including detailed analysis of trends and unique perspectives"

def extract_links_from_text(text: str) -> list[str]:
    """Finds all URLs (http, https, or /r/ paths) in a piece of text and cleans them."""
    # Find absolute URLs
    urls = re.findall(r'(https?://\S+)', text)
    # Find relative Reddit URLs
    reddit_paths = re.findall(r'(/r/\S+)', text)
    
    cleaned_urls = []
    for url in urls + reddit_paths:
        # Remove trailing punctuation, parentheses, and brackets
        url = re.sub(r'[.,!?\]\[()]+$', '', url)
        url = normalize_reddit_url(url)  # Convert relative URLs to absolute
        cleaned_urls.append(url)
    return cleaned_urls

def is_image_link(url: str) -> bool:
    return any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])

def is_video_link(url: str) -> bool:
    return any(ext in url.lower() for ext in ['.mp4', '.mov', '.avi', 'v.redd.it', 'youtube.com', 'youtu.be'])

def clean_html_content(html_content: str) -> str:
    """
    Takes raw HTML content and returns cleaned text version.
    """
    try:
        # Use readability to extract main content
        doc = Document(html_content)
        
        # Get both title and content
        title = doc.title()
        content = doc.summary()
        
        # Clean the HTML content
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text and clean up whitespace
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n\n".join(lines)
        
        # Combine title and content
        if title:
            return f"Title: {title}\n\n{cleaned_text}"
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Failed to clean HTML content: {str(e)}")
        return ""

def fetch_clean_html_content(url: str) -> str:
    """
    Downloads the given URL, uses readability-lxml to identify the main article content,
    and returns a cleaned text version of that content.
    """
    logger.info(f"Fetching content from {url}")
    try:
        # Check if the content is raw HTML (starts with <!DOCTYPE or <html)
        if url.strip().lower().startswith(("<!doctype", "<html")):
            return clean_html_content(url)
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        return clean_html_content(response.text)
        
    except Exception as e:
        logger.error(f"Failed to fetch URL: {url} => {str(e)}")
        raise RuntimeError(f"Failed to fetch URL: {url} => {str(e)}")

def create_summary_prompt(text: str, length_guidance: str) -> PromptTemplate:
    """
    Creates a dynamic prompt template for text summarization 
    with length guidance (short, medium, or long).
    """
    return PromptTemplate(
        input_variables=["text", "length_guidance"],
        template="""Provide a detailed summary of the following text.

Length Requirement: {length_guidance}

Guidelines:
- Maintain all key points and important details
- Use clear topic sentences
- For longer summaries, use bullet points or clear paragraph breaks
- Include specific examples or data points when present

Text to summarize:
{text}"""
    )

def summarize_text_with_llm(text: str) -> str:
    """
    Summarizes 'text' using a prompt that adapts the length to the text size.
    """
    if not text.strip():
        return "No content to summarize."
    try:
        word_count = get_word_count(text)
        length_guidance = get_summary_length_guidance(word_count)
        chain = create_summary_prompt(text, length_guidance) | llm
        summary = chain.invoke({"text": text, "length_guidance": length_guidance})
        return summary.content.strip()
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"

def summarize_link_briefly(text: str) -> str:
    """
    Produces a concise 2-3 sentence summary for external links or video transcripts.
    """
    if not text.strip():
        return "No content to summarize."
    
    short_link_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Please provide a concise 2-3 sentence summary of this content. 

{text}

Start your response with "The article discusses" or similar introductory phrase if it's an article. 
If it is a youtube link then start your response with "The youtube video shows..." or a similar phrase.
"""
    )
    
    try:
        chain = short_link_prompt | llm
        response = chain.invoke({"text": text})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Link summarization error: {str(e)}")
        return f"An error occurred during link summarization: {str(e)}"

########################
# OCR & YOUTUBE SUPPORT
########################

def extract_text_from_image(url: str) -> str:
    """
    Downloads an image from `url` and performs OCR to extract text.
    Requires Tesseract installed on your system.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img_data = response.content
        image = Image.open(BytesIO(img_data))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.error(f"Image OCR error for {url}: {str(e)}")
        return ""

def get_video_id(youtube_url: str) -> str:
    """
    Extracts the video_id from typical youtube.com or youtu.be URLs.
    """
    try:
        parsed = urlparse(youtube_url)
        if parsed.hostname == 'youtu.be':
            return parsed.path[1:]
        elif parsed.hostname and 'youtube.com' in parsed.hostname:
            qs = parse_qs(parsed.query)
            return qs.get('v', [None])[0]
        return None
    except Exception as e:
        logger.error(f"Failed to extract YouTube video ID from {youtube_url}: {str(e)}")
        return None

def summarize_youtube_link(url: str) -> str:
    """
    Fetches YouTube transcript and provides a summary if available.
    If no transcript is found, returns a clear message.
    """
    video_id = get_video_id(url)
    if not video_id:
        return "Could not process YouTube video URL."
    
    try:
        # First try to get available transcript languages
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try English first
            try:
                transcript = transcript_list.find_transcript(['en'])
                transcript_text = " ".join(seg['text'] for seg in transcript.fetch())
            except:
                # If no English, try auto-generated
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                    transcript_text = " ".join(seg['text'] for seg in transcript.fetch())
                except:
                    # If no English at all, get the first available language
                    transcript = transcript_list.find_manually_created_transcript()
                    transcript_text = " ".join(seg['text'] for seg in transcript.fetch())
        except:
            # If listing fails, try direct fetch (legacy approach)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join(seg['text'] for seg in transcript_list)

        if not transcript_text.strip():
            return "No transcript found for this YouTube video."
            
        return summarize_link_briefly(transcript_text)
    except Exception as e:
        logger.error(f"Failed to get transcript for YouTube video {url}: {str(e)}")
        return "No transcript found for this YouTube video."

def summarize_image_link(url: str) -> str:
    """
    Extracts text via OCR and summarizes it if any text is found.
    """
    ocr_text = extract_text_from_image(url)
    if not ocr_text:
        return f"[Image Link: {url}, Image not found or no text detected]."
    summary = summarize_link_briefly(ocr_text)
    return summary  # Return just the summary

##############################
# CHUNKING FOR LARGE VIDEOS
##############################

def extract_video_frames_in_range(video_path: str, start_sec: float, end_sec: float, frame_interval=30) -> list[str]:
    """
    Extract frames from 'video_path' between 'start_sec' and 'end_sec',
    grabbing 1 frame every 'frame_interval' frames or so.
    Return a list of base64-encoded images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    frame_count = 0
    base64_frames = []
    current_timestamp = start_sec
    
    while True:
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_timestamp > end_sec:
            break
        success, frame = cap.read()
        if not success:
            break
        
        # Grab every Nth frame
        if (frame_count % frame_interval) == 0:
            success_jpg, buffer = cv2.imencode(".jpg", frame)
            if success_jpg:
                encoded_str = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(encoded_str)
        frame_count += 1

    cap.release()
    return base64_frames

def get_video_duration(video_path: str) -> float:
    """Return the total duration (in seconds) of the given video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

def summarize_frames_with_llm(base64_frames: list[str], chunk_label="Segment") -> str:
    """
    Summarize frames by sending them to GPT in one request.
    If you have too many frames, sample them further.
    """
    frames_to_send = base64_frames[:10]  # Limit to first 10 frames
    user_content = [
        f"These are frames from the video {chunk_label}. Please describe what's happening in them.",
    ]
    for frame_b64 in frames_to_send:
        user_content.append({"image": frame_b64, "resize": 768})

    messages = [{"role": "user", "content": user_content}]
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content

def summarize_video_in_chunks(video_path: str, chunk_minutes=5) -> str:
    """
    Break a local video into intervals, extract frames, summarize each chunk,
    then do a final summary pass.
    """
    duration = get_video_duration(video_path)
    if duration == 0.0:
        return "Error: Could not read video duration."

    chunk_duration_sec = chunk_minutes * 60
    start_sec = 0.0
    sub_summaries = []
    chunk_index = 1

    while start_sec < duration:
        end_sec = min(start_sec + chunk_duration_sec, duration)
        frames = extract_video_frames_in_range(video_path, start_sec, end_sec, frame_interval=30)
        chunk_summary = summarize_frames_with_llm(frames, chunk_label=f"Chunk {chunk_index}")
        sub_summaries.append(f"**Chunk {chunk_index} Summary:** {chunk_summary}")
        start_sec += chunk_duration_sec
        chunk_index += 1

    combined_sub_summaries = "\n\n".join(sub_summaries)
    final_summary_prompt = f"Summarize these chunk summaries:\n\n{combined_sub_summaries}"
    combined_summary_resp = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": final_summary_prompt}],
        max_tokens=500
    )
    final_summary = combined_summary_resp.choices[0].message.content
    return f"**All Chunk Summaries**:\n\n{combined_sub_summaries}\n\n**Final Merged Summary**:\n\n{final_summary}"

############################
# CLEAN PARAGRAPH SPACING
############################

def clean_paragraph_spacing(text: str) -> str:
    """
    Converts text to AsciiDoc format and cleans up spacing.
    """
    # Split into lines and process
    lines = text.split('\n')
    formatted_lines = []
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Convert headers to AsciiDoc format
        if stripped_line.startswith(('**', '#')):
            level = stripped_line.count('#') if '#' in stripped_line else 1
            clean_text = stripped_line.replace('**', '').replace('#', '').strip()
            formatted_lines.append(f"{'=' * level} {clean_text}")
        # Convert bullet points to AsciiDoc format
        elif stripped_line.startswith(('- ', '* ', '+ ')):
            if i > 0 and not lines[i-1].strip().startswith(('- ', '* ', '+ ')):
                formatted_lines.append('')
            bullet_content = stripped_line[2:].strip()
            formatted_lines.append(f"* {bullet_content}")
        else:
            formatted_lines.append(line)
    
    text = '\n'.join(formatted_lines)
    
    # Clean up excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Convert other formatting to AsciiDoc
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)  # Bold
    text = re.sub(r'`(.*?)`', r'+\1+', text)        # Monospace
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1 (\2)', text)  # Links
    
    return text.strip()

########################################
# ---- Summarization FastAPI Routes ----
########################################

def normalize_reddit_url(url: str) -> str:
    """
    Converts relative Reddit URLs to absolute URLs.
    """
    if url.startswith('/r/'):
        return f'https://reddit.com{url}'
    return url

@app.get("/summarize")
async def summarize_post(url: str = Query(..., description="URL of the Reddit post")):
    """
    Summarize a Reddit post + top comments. Also handle external links or images in the post.
    """
    try:
        logger.info(f"Received URL: {url}")
        
        # Normalize the URL if it's a relative Reddit URL
        url = normalize_reddit_url(url)

        # Verify Reddit credentials
        try:
            await reddit.user.me()
        except Exception as e:
            logger.error(f"Reddit authentication error: {str(e)}")
            return {"error": "Reddit authentication failed. Please check your credentials."}

        # Fetch submission using AsyncPRAW
        try:
            submission = await reddit.submission(url=url)
            await submission.load()
            logger.info(f"Successfully fetched submission: {submission.title}")
        except asyncprawcore.exceptions.NotFound:
            return {"error": "Reddit post not found or it may be deleted/archived."}
        except asyncprawcore.exceptions.Forbidden:
            return {"error": "This Reddit post is private or restricted."}
        except (asyncprawcore.exceptions.ResponseException,
                asyncprawcore.exceptions.RequestException) as e:
            logger.error(f"Reddit API error details: {str(e)}")
            return {"error": f"Reddit API error: {str(e)}"}

        # Handle cross-posts and linked posts
        linked_post_content = ""
        if hasattr(submission, 'crosspost_parent_list') and submission.crosspost_parent_list:
            try:
                # Get the original post from crosspost
                original_submission = await reddit.submission(id=submission.crosspost_parent_list[0]['id'])
                await original_submission.load()
                linked_post_content = (
                    f"\nOriginal Post Content:\n"
                    f"{original_submission.title}\n{original_submission.selftext}"
                )
            except Exception as e:
                logger.error(f"Error fetching crosspost: {str(e)}")
        elif submission.url and 'reddit.com/r/' in submission.url and not submission.is_self:
            try:
                # Extract submission ID from the URL and fetch the linked post
                linked_url = submission.url
                linked_submission = await reddit.submission(url=linked_url)
                await linked_submission.load()
                linked_post_content = (
                    f"\nLinked Post Content:\n"
                    f"{linked_submission.title}\n{linked_submission.selftext}"
                )
            except Exception as e:
                logger.error(f"Error fetching linked post: {str(e)}")

        # 1) Collect main post content
        post_content_parts = []
        if submission.title:
            post_content_parts.append(submission.title)
            
        # Check for poll results first
        poll_results = get_poll_results(submission)
        if poll_results:
            post_content_parts.append(poll_results)
            if not submission.selftext and not submission.url:
                return {"Summary": poll_results}
                
        if submission.selftext:
            post_content_parts.append(submission.selftext)
        if linked_post_content:
            post_content_parts.append(linked_post_content)

        # ----------------------------------------------------------
        # NEW - Image Post Handling
        # ----------------------------------------------------------
        # If the submission itself is an image post, or if the URL is from a known
        # image host (e.g. i.redd.it, i.imgur.com, etc.), perform OCR
        # (and skip fetch_clean_html_content).
        # You can check submission.post_hint == "image" or submission.is_gallery.
        # Also, we can do a small 'domain' check for i.redd.it, i.imgur.com, etc.
        # ----------------------------------------------------------

        def is_image_domain(link: str) -> bool:
            return any(
                domain in link.lower()
                for domain in ["i.redd.it", "i.imgur.com", "ibb.co", "flickr.com"]
            )

        image_ocr_handled = False

        if (
            getattr(submission, "post_hint", "") == "image"
            or is_image_domain(submission.url)
            # or submission.preview might exist with images
        ):
            logger.info("Detected an image post. Attempting OCR.")

            # Single image case
            if not getattr(submission, "is_gallery", False):
                ocr_text = extract_text_from_image(submission.url)
                if ocr_text:
                    summary = summarize_link_briefly(ocr_text)
                    post_content_parts.append(f"\n[Image OCR Summary]\n{summary}")
                else:
                    post_content_parts.append("[Image was detected, but no text was found (or OCR failed).]")
                image_ocr_handled = True

            # If it's a gallery/album of multiple images
            elif getattr(submission, "is_gallery", False):
                logger.info("Detected a gallery post. Attempting OCR on each image in the gallery.")
                media_metadata = submission.media_metadata
                if not media_metadata:
                    post_content_parts.append("[No accessible media metadata for gallery]")
                else:
                    gallery_summaries = []
                    for media_id, media_data in media_metadata.items():
                        # Each media_data typically has 'p', 's', etc. containing sources
                        # We'll pick the largest or original source
                        source = media_data.get('s', {})
                        image_url = source.get('u') or source.get('gif')
                        if image_url:
                            # Reddit often replaces some characters. 
                            # Clean up ampersands, e.g. &amp; => &
                            image_url = image_url.replace("&amp;", "&")

                            ocr_text = extract_text_from_image(image_url)
                            if ocr_text:
                                single_image_summary = summarize_link_briefly(ocr_text)
                                gallery_summaries.append(f"- **Gallery Item**: {single_image_summary}")
                            else:
                                gallery_summaries.append("- [No text found in this gallery image]")
                    
                    if gallery_summaries:
                        joined_summaries = "\n".join(gallery_summaries)
                        post_content_parts.append(f"\n[Gallery OCR Summaries]\n{joined_summaries}")
                    else:
                        post_content_parts.append("[Gallery was detected, but no images or text found].")
                image_ocr_handled = True

        # If not an image post, handle external URL as an article or skip
        if submission.url and not submission.is_self and not image_ocr_handled:
            try:
                # Skip summarizing links that lead back to reddit.com or /r/
                if ('reddit.com' not in submission.url.lower()) and ('/r/' not in submission.url.lower()):
                    normalized_url = normalize_reddit_url(submission.url)
                    article_content = fetch_clean_html_content(normalized_url)
                    article_summary = summarize_link_briefly(article_content)
                    post_content_parts.append(f"\nLinked Article Summary:\n{article_summary}")
                else:
                    # It's a Reddit link; skip external summarization
                    pass
            except Exception as e:
                logger.error(f"Error processing URL {submission.url}: {str(e)}")
                post_content_parts.append(f"[External Link: {submission.url}]")

        # Check for video content
        video_content = ""
        if hasattr(submission, 'media') and submission.media:
            try:
                # Handle Reddit hosted videos
                if submission.is_video:
                    video_url = submission.media['reddit_video']['fallback_url']
                    logger.info(f"Found Reddit video: {video_url}")
                    video_content = process_video_content(video_url)
                    if video_content:
                        post_content_parts.append(video_content)
                        logger.info("Added video transcript to content")
                
                # Handle other video platforms (v.redd.it, etc.)
                elif 'v.redd.it' in submission.url:
                    video_content = process_video_content(submission.url)
                    if video_content:
                        post_content_parts.append(video_content)
                        logger.info("Added external video transcript to content")
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                post_content_parts.append(f"[Video content available but could not be processed: {str(e)}]")

        # Combine all content
        full_content = "\n\n".join(post_content_parts)

        # 2) Summarize the main post text
        post_summary = summarize_text_with_llm(full_content)

        # 2b) Summarize any embedded links inside the text
        embedded_links = extract_links_from_text(full_content)
        article_context_summaries = []
        for link in embedded_links:
            try:
                if is_image_link(link) or is_image_domain(link):
                    summary = summarize_image_link(link)
                    article_context_summaries.append(("Image Info", summary))
                elif 'youtube.com' in link or 'youtu.be' in link:
                    summary = summarize_youtube_link(link)
                    article_context_summaries.append(("YouTube Summary", summary))
                elif is_video_link(link):
                    article_context_summaries.append(("Video Content", f"[Video Link: {link}]"))
                else:
                    # Skip if it's a reddit link
                    if ('reddit.com' not in link.lower()) and ('/r/' not in link.lower()):
                        text_content = fetch_clean_html_content(link)
                        brief_summary = summarize_link_briefly(text_content)
                        article_context_summaries.append(("Article Context", brief_summary))
            except Exception as e:
                logger.error(f"Could not fetch/parse link {link}: {str(e)}")
                article_context_summaries.append(("Error", f"[Unable to summarize link: {link}]"))

        # 3) Summarize top comments
        await submission.comments.replace_more(limit=0)
        top_comments = list(submission.comments)[:15]
        comment_texts = []
        total_comment_words = 0
        for comment in top_comments:
            await comment.load()
            comment_texts.append(comment.body)
            total_comment_words += get_word_count(comment.body)
        combined_comments = "\n\n---\n\n".join(comment_texts)

        length_guidance = get_comment_length_guidance(len(comment_texts), total_comment_words)
        dynamic_comment_prompt = PromptTemplate(
            input_variables=["comments", "length_guidance"],
            template="""Analyze these Reddit comments and provide a clear analysis.

Length Requirement: {length_guidance}

Focus on:
- The main consensus/prevalent opinions
- Notable outlier perspectives
- Specific examples or unique insights
- Emerging patterns or trends

Comments to analyze:
{comments}"""
        )
        chain = dynamic_comment_prompt | llm
        comment_analysis = chain.invoke({
            "comments": combined_comments,
            "length_guidance": length_guidance
        })
        comments_summary = comment_analysis.content.strip()

        # 4) Combine it into a final summary
        final_summary_parts = []
        final_summary_parts.append(f"**Post Summary:**\n\n{post_summary.strip()}\n")

        # Only include article context summaries if there are actual articles
        article_summaries = [
            (header, content) for header, content in article_context_summaries 
            if header == "Article Context" and not content.startswith("[Unable to summarize")
        ]
        if article_summaries:
            for header, content in article_summaries:
                final_summary_parts.append(f"**{header}:**\n\n{content.strip()}\n")
        
        # Include other non-article summaries (YouTube, Images, etc)
        other_summaries = [
            (header, content) for header, content in article_context_summaries 
            if header != "Article Context" and not content.startswith("[Unable to summarize")
        ]
        for header, content in other_summaries:
            final_summary_parts.append(f"**{header}:**\n\n{content.strip()}\n")
        
        final_summary_parts.append(f"**What People Said:**\n\n{comments_summary}")

        draft_summary = "\n\n".join(final_summary_parts)
        cleaned_summary = clean_paragraph_spacing(draft_summary)
        return {"Summary": cleaned_summary}

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


@app.get("/test-reddit")
async def test_reddit_connection():
    """
    Quick check to see if Reddit credentials are working.
    """
    try:
        await reddit.user.me()
        return {"status": "success", "message": "Reddit credentials are working"}
    except Exception as e:
        logger.error(f"Reddit authentication test failed: {str(e)}")
        return {"status": "error", "message": f"Reddit authentication failed: {str(e)}"}

@app.on_event("shutdown")
async def cleanup():
    """
    Make sure to close the reddit client when the app stops.
    """
    await reddit.close()

def make_openai_request(attempt=1, max_attempts=3):
    """
    Example helper for gracefully handling OpenAI requests.
    """
    try:
        return openai.ChatCompletion.create(...)  # Replace with your parameters
    except OpenAIError as e:
        if "insufficient_quota" in str(e):
            print("API quota exceeded. Check your usage at platform.openai.com")
            return None
        elif attempt < max_attempts:
            wait_time = min(2 ** attempt, 8)
            time.sleep(wait_time)
            return make_openai_request(attempt + 1, max_attempts)
        else:
            raise e

def get_poll_results(submission) -> str:
    """
    Extract poll results from a Reddit submission if it has a poll.
    Returns empty string if no poll is found or no data is available.
    """
    try:
        # Check for poll_data attribute
        if not hasattr(submission, 'poll_data') or not submission.poll_data:
            return ""

        poll_data = submission.poll_data
        total_votes = poll_data.total_vote_count
        if total_votes == 0:
            return "Poll has no votes yet."

        # Gather options and their counts
        options = []
        for option in poll_data.options:
            vote_percentage = (option.votes / total_votes) * 100
            options.append(
                f"- {option.text}: {vote_percentage:.1f}% ({option.votes} votes)"
            )

        # Sort options by descending vote count
        options.sort(
            key=lambda opt: float(re.search(r"\((\d+) votes\)", opt).group(1)),
            reverse=True
        )

        results = [f"Poll Results (Total Votes: {total_votes})"]
        results.extend(options)
        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error processing poll data: {str(e)}")
        return ""

def download_video(url: str) -> str:
    """
    Downloads a video from a URL and returns the local path.
    """
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'temp_video.mp4')
        
        # Download the video
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return None

def extract_audio_from_video(video_path: str) -> str:
    """
    Extracts audio from video and returns path to audio file.
    """
    try:
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, 'temp_audio.wav')
        
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio file to text using speech recognition.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None

def process_video_content(video_url: str) -> str:
    """
    Process video content: download, extract audio, and transcribe.
    """
    try:
        logger.info(f"Processing video from URL: {video_url}")
        
        # Download video
        video_path = download_video(video_url)
        if not video_path:
            return "[Error: Could not download video]"
            
        # Extract audio
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return "[Error: Could not extract audio from video]"
            
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return "[Error: Could not transcribe video audio]"
            
        # Clean up temporary files
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except:
            pass
            
        return f"Video Transcript:\n{transcript}"
    except Exception as e:
        logger.error(f"Error processing video content: {str(e)}")
        return f"[Error processing video: {str(e)}]"

############################
# MAIN (if you want to run)
############################
if __name__ == "__main__":
    # Run on port 8000 by default
    uvicorn.run(app, host="0.0.0.0", port=8000)
