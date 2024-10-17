import os
import asyncio
import praw
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import openai  # Import the OpenAI library
import pandas as pd
from pydantic.json import ENCODERS_BY_TYPE

# Add DataFrame to Pydantic's JSON encoders
ENCODERS_BY_TYPE[pd.DataFrame] = lambda df: df.to_dict(orient="records")

app = FastAPI()
load_dotenv()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your extension's origin or allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_API_KEY"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

def summarize_text(text):
    if not text.strip():
        return "No content to summarize."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please provide a concise summary of the following text:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

@app.get("/summarize")
async def summarize_post(url: str = Query(..., description="URL of the Reddit post")):
    # Extract the post ID from the URL
    post_id = url.rstrip('/').split('/')[-3]
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)

    # Summarize the post content
    post_summary = summarize_text(submission.selftext or submission.title)

    # Summarize the top 10 comments
    comment_summaries = []
    for comment in submission.comments[:10]:
        comment_summary = summarize_text(comment.body)
        comment_summaries.append(comment_summary)

    # Construct the final summary
    final_summary = f"Title: {submission.title}\n\nPost Summary:\n{post_summary}\n\nTop Replies:\n"
    for i, summary_text in enumerate(comment_summaries, 1):
        final_summary += f"{i}. {summary_text}\n"

    return {"Summary": final_summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
