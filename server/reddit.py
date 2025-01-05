import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Query
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

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://dhgjompomnkacadcomfjemhancdmdkmd"],
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
llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0.5, openai_api_key=openai_api_key)

# Define the prompt template for summarization
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Please provide a summary of the following text. The summary length should be proportional to the original text length:
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

# Simpler alternative if the above doesn't work
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize Reddit API client
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_API_KEY"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

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

# Update the summarize_text function
def summarize_text(text: str) -> str:
    if not text.strip():
        return "No content to summarize."
    try:
        return _extracted_from_summarize_text_5(text)
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"


# TODO Rename this here and in `summarize_text`
def _extracted_from_summarize_text_5(text):
    word_count = get_word_count(text)
    length_guidance = get_summary_length_guidance(word_count)

    dynamic_prompt = PromptTemplate(
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

    chain = dynamic_prompt | llm
    summary = chain.invoke({"text": text, "length_guidance": length_guidance})
    return summary.content.strip()

@app.get("/summarize")
async def summarize_post(url: str = Query(..., description="URL of the Reddit post")):
    try:
        logger.info(f"Received URL: {url}")

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
            return {"error": "Reddit post not found. It may have been deleted or archived."}
        except asyncprawcore.exceptions.Forbidden:
            return {"error": "Cannot access this Reddit post. It may be private or restricted."}
        except (asyncprawcore.exceptions.ResponseException, 
                asyncprawcore.exceptions.RequestException) as e:
            logger.error(f"Reddit API error details: {str(e)}")
            return {"error": f"Reddit API error: {str(e)}"}

        # Get post content including media
        post_content = ""
        if hasattr(submission, 'selftext') and submission.selftext:
            post_content = submission.selftext
        elif hasattr(submission, 'url') and submission.url:
            if submission.is_self:
                post_content = submission.title
            else:
                # Handle media posts
                post_content = f"{submission.title}\n\nPost contains: "
                if 'imgur' in submission.url or submission.url.endswith(('.jpg', '.png', '.gif')):
                    post_content += f"[Image: {submission.url}]"
                elif 'youtube' in submission.url or 'youtu.be' in submission.url:
                    post_content += f"[Video: {submission.url}]"
                else:
                    post_content += f"[Link: {submission.url}]"

        post_summary = summarize_text(post_content)

        # Gather and analyze comments with length awareness
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

        # Format the final summary with clear structure
        final_summary = f"# Post Summary:\n{post_summary.strip()}\n\n# What People Said:\n{comment_analysis.content.strip()}"
        
        return {"Summary": final_summary}
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

@app.get("/test-reddit")
async def test_reddit_connection():
    try:
        await reddit.user.me()
        return {"status": "success", "message": "Reddit credentials are working"}
    except Exception as e:
        logger.error(f"Reddit authentication test failed: {str(e)}")
        return {"status": "error", "message": f"Reddit authentication failed: {str(e)}"}

# Add this cleanup code to properly close the session
@app.on_event("shutdown")
async def cleanup():
    await reddit.close()

def make_openai_request(attempt=1, max_attempts=3):
    try:
        return openai.ChatCompletion.create(...)  # Replace ... with your actual parameters
    except OpenAIError as e:
        if "insufficient_quota" in str(e):
            print("API quota exceeded. Please check your billing status at platform.openai.com")
            return None
        elif attempt < max_attempts:
            wait_time = min(2 ** attempt, 8)
            time.sleep(wait_time)
            return make_openai_request(attempt + 1, max_attempts)
        else:
            raise e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
