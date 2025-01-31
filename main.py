import time
from typing import List
from dotenv import load_dotenv
import os

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from prompt_templates import template_default
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate
import os
import pprint
from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import OpenAI

from langchain_community.utilities import GoogleSerperAPIWrapper

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
import asyncio
import json
import time

from utils.misc import measure_time
import asyncio
import json

# Initialize
app = FastAPI()
logger = None
agent_with_chat_history = None
next_start_time = None

@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()

    load_dotenv(".env")
    BASE_URL = os.getenv("BASE_URL")
    MODEL_NAME = os.getenv['MODEL_NAME']
    TEMPERATURE = os.getenv['TEMPERATURE']
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

    search = GoogleSerperAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.results,
            description="Useful when need to ask with search",
        ),
    ]

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        base_url=BASE_URL,
        max_tokens=None,
        timeout=None,
    )

    memory = ChatMessageHistory(session_id="test-session")
    prompt = PromptTemplate.from_template(template_default)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=True
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="query",
        history_messages_key="chat_history"
    )

    next_start_time = asyncio.get_event_loop().time()  # Initialize next start time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

max_async_queries = 20
semaphore = asyncio.Semaphore(max_async_queries) if max_async_queries else None
start_lock = asyncio.Lock()  # Lock to manage sequential start delays

@measure_time
async def async_inference(agent_with_chat_history, query):
    response = await agent_with_chat_history.ainvoke(
        {"query": query},
        config={
            "configurable": {"session_id": "test-session"},
            # "tracing": True
        }
    )
    return json.loads(response["output"])

RATE_LIMIT_S = 1.1

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        global next_start_time  # Access shared start time

        while True:
            async with start_lock:  # Ensure sequential access to next_start_time
                now = asyncio.get_event_loop().time()
                if now >= next_start_time:
                    next_start_time = now + RATE_LIMIT_S  # Schedule next task start time
                    break  # Proceed if it's time to start

            await asyncio.sleep(RATE_LIMIT_S)  # Otherwise, halt and check again

        async with semaphore:  # Ensure limited concurrency
            output, iter_time = await async_inference(agent_with_chat_history, query["query"])
            print(f"Iter time: {iter_time}")

            response = PredictionResponse(
                id=body.id,
                **output
            )

            await logger.info(f"Successfully processed request {body.id}")
            return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
