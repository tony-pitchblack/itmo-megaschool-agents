import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger, flush_aiologger
import os
import glob

from aiologger.levels import LogLevel
from prompt_templates import template_default

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
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
import json

# Initialize
app = FastAPI()
logger = None

# TODO: load from dotenv
BASE_URL="https://api.vsegpt.ru/v1"
API_KEY="sk-or-vv-ac07a84e938358eb29c38b15da3a139c55c49a467c221d885d8c20aad7c2e62f"
OPENAI_API_KEY="sk-or-vv-ac07a84e938358eb29c38b15da3a139c55c49a467c221d885d8c20aad7c2e62f"
SERPER_API_KEY="777ae7587f22cab185f347439bff7e4639cea5ae"
LANGSMITH_TRACING="true"
LANGCHAIN_TRACING_V2="true"
LANGSMITH_API_KEY="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"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# MODEL_NAME="gpt-4o-latest"
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.1

async def clear_logs(log_folder = "./logs/"):
    # Get a list of all files in the logs directory
    log_files = glob.glob(os.path.join(log_folder, "*"))

    # Delete each file
    for file_path in log_files:
        try:
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# async def setup_env_vars():
#     global logger
#     load_dotenv('/home/admin/itmo-megaschool-agents/.env')

#     await logger.debug(f"All Environment Variables: {os.environ}")

#     env_vars = {
#         'BASE_URL': os.getenv("BASE_URL"),
#         'SERPER_API_KEY': os.getenv("SERPER_API_KEY"),
#         "OPENAI_API_KEY": os.getenv("API_KEY")
#     }

#     await logger.debug(f"Loaded env vars: {env_vars}")
#     return env_vars

agent_with_chat_history = None

@app.on_event("startup")
async def startup_event():
    global logger
    global agent_with_chat_history

    logger = await setup_logger(LogLevel.DEBUG)

    await clear_logs()
    await logger.info('Startup successful')
    await logger.debug(f"Script Directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # env_vars = await setup_env_vars()

    prompt = PromptTemplate.from_template(template_default)
    await logger.debug(prompt)

    search = GoogleSerperAPIWrapper(
        serper_api_key=SERPER_API_KEY
    )

    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.results,
            description="Useful when need to ask with search",
        ),
    ]
    await logger.debug('Set up tools')

    llm = ChatOpenAI(
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        base_url=BASE_URL,
        max_tokens=None,
        timeout=None,
    )
    await logger.debug("Set up llm")

    memory = ChatMessageHistory(session_id="test-session")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence=True
    )
    await logger.debug("Set up agent")

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    await logger.debug("Set up agent_executor")

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="query",
        history_messages_key="chat_history"
    )
    await logger.debug('Set up agent_with_chat_history')


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

async def inference(query):
    global logger

    await logger.debug("Running inference")
    response = agent_with_chat_history.invoke(
        {"query": query},
        config={
            "configurable": {"session_id": "test-session"},
        }
    )
    await logger.debug("Successful inference")

    output = json.loads(response['output'])
    return output

@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    logger.info('hello world')
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        output = await inference(body.query)

        # output = {
        #     "answer": 0,
        #     "reasoning": '',
        #     "sources": ["https://www.google.com/"]
        # }

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
