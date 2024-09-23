import praw
import os
from langchain_openai import OpenAI  # Updated import
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

api_key = os.getenv('OPENAI_API_KEY')

class RedditUrl(BaseModel):
    url: str

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_API_KEY"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

llm = OpenAI(temperature=0.5)

chain = load_summarize_chain(llm, chain_type="map_reduce")

def summarize_text(text):
    docs = [Document(page_content=text)]
    summary = chain.run(docs)
    return summary

@app.post("/summarize")
async def summarize_post(reddit_url: RedditUrl):
    url = reddit_url.url

    post_id = url.split('/')[-3]

    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)

    post_summary = summarize_text(submission.selftext)
    
    comment_summaries = []
    for comment in submission.comments[:10]:
        comment_summary = summarize_text(comment.body)
        comment_summaries.append(comment_summary)
    
    final_summary = f"Title: {submission.title}\n\n{post_summary}\n\nReplies:\n\n"
    for i, summary_text in enumerate(comment_summaries, 1):
        final_summary += f"{i}. {summary_text}\n"
    
    return final_summary

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
