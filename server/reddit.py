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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)

# Define the prompt template for summarization
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary of the following text:\n\n{text}"
)

# Simpler alternative if the above doesn't work
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize Reddit API client
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_API_KEY"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def summarize_text(text: str) -> str:
    if not text.strip():
        return "No content to summarize."
    try:
        chain = summary_prompt | llm
        summary = chain.invoke({"text": text})
        return summary.content.strip()
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"

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
            # Force load the submission attributes
            await submission.load()  # This loads all attributes at once
            logger.info(f"Successfully fetched submission: {submission.title}")
        except asyncprawcore.exceptions.NotFound:
            return {"error": "Reddit post not found. It may have been deleted or archived."}
        except asyncprawcore.exceptions.Forbidden:
            return {"error": "Cannot access this Reddit post. It may be private or restricted."}
        except (asyncprawcore.exceptions.ResponseException, 
                asyncprawcore.exceptions.RequestException) as e:
            logger.error(f"Reddit API error details: {str(e)}")
            return {"error": f"Reddit API error: {str(e)}"}

        # Summarize the post content
        post_content = submission.selftext if hasattr(submission, 'selftext') else submission.title
        post_summary = summarize_text(post_content)
        logger.debug(f"Post summary: {post_summary}")

        # Summarize the top 10 comments
        comment_summaries = []
        await submission.comments.replace_more(limit=0)  # Flatten comment tree
        top_comments = list(submission.comments)[:10]
        for comment in top_comments:
            await comment.load()  # Load comment attributes
            summary = summarize_text(comment.body)
            comment_summaries.append(summary)
            logger.debug(f"Comment summary: {summary}")

        # Construct the final summary
        final_summary = f"**Title:** {submission.title}\n\n**Post Summary:**\n{post_summary}\n\n**Top Replies:**\n"
        for i, summary in enumerate(comment_summaries, 1):
            final_summary += f"{i}. {summary}\n"

        logger.debug("Final summary constructed")
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
