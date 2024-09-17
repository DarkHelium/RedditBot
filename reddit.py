import praw
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
app = FastAPI()

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


chain = LLMChain(llm=llm, chain_type="map_reduce")

def summarize_text(text):
    doc = Document(page_content=text)
    summary = chain.run([doc])
    return summary

@app.post("/summarize")
async def summarize_post(reddit_url:RedditUrl):
    url = reddit_url.url

    post_id = url.split('/')[-3]

    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)

    post_summary = summarize_text(submission.selftext)
    
    comment_summaries = []
    for comment in submission.comments[:10]:
        comment_summary = summarize_text(comment.body)
        comment_summaries.append(comment_summary)
    
    summary = f"Title: {submission.title}\n\n {post_summary} \n\n Replies: \n\n"
    for i, summary in enumerate(comment_summaries, 1):
        final_summary += f"{i}. {summary}\n"
    
    return final_summary

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)