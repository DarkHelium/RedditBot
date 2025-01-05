import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import openai
import logging
import praw

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Initialize the LLMChain for summarization
summarizer = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    output_key="summary"
)

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def summarize_text(text: str) -> str:
    if not text.strip():
        return "No content to summarize."
    try:
        summary = summarizer.run(text)
        return summary.strip()
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"An error occurred during summarization: {str(e)}"

@app.get("/summarize")
async def summarize_post(url: str = Query(..., description="URL of the Reddit post")):
    try:
        logger.debug(f"Received URL: {url}")

        # Fetch submission using PRAW
        submission = reddit.submission(url=url)
        logger.debug(f"Fetched submission: {submission.title}")

        # Summarize the post content
        post_summary = summarize_text(submission.selftext or submission.title)
        logger.debug(f"Post summary: {post_summary}")

        # Summarize the top 10 comments
        comment_summaries = []
        submission.comments.replace_more(limit=0)  # Flatten comment tree
        top_comments = list(submission.comments)[:10]
        for comment in top_comments:
            summary = summarize_text(comment.body)
            comment_summaries.append(summary)
            logger.debug(f"Comment summary: {summary}")

        # Construct the final summary
        final_summary = f"**Title:** {submission.title}\n\n**Post Summary:**\n{post_summary}\n\n**Top Replies:**\n"
        for i, summary in enumerate(comment_summaries, 1):
            final_summary += f"{i}. {summary}\n"

        logger.debug("Final summary constructed")
        return {"Summary": final_summary}
    except praw.exceptions.PRAWException as e:
        logger.error(f"Reddit API error: {str(e)}")
        return {"error": f"Reddit API error: {str(e)}"}
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
