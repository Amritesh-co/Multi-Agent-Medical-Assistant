#!/usr/bin/env python3
"""Debug script to test decision chain without full app."""

import os
from dotenv import load_dotenv
load_dotenv()

from config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict

config = Config()
decision_model = config.agent_decision.llm

class AgentDecision(TypedDict):
    agent: str
    reasoning: str
    confidence: float

json_parser = JsonOutputParser(pydantic_object=AgentDecision)

DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
is best suited to handle it based on the query content, presence of images, and conversation context.

Available agents:
1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
2. RAG_AGENT - For specific medical knowledge questions.

You must provide your answer in JSON format with the following structure:
{{
"agent": "AGENT_NAME",
"reasoning": "Your step-by-step reasoning for selecting this agent",
"confidence": 0.95
}}
"""

decision_prompt = ChatPromptTemplate.from_messages([
    ("system", DECISION_SYSTEM_PROMPT),
    ("human", "{input}")
])

decision_chain = decision_prompt | decision_model | json_parser

# Test query
test_input = """
User query: Hello, how are you?

Recent conversation context:

Has image: False
Image type: None

Based on this information, which agent should handle this query?
"""

print("Testing decision chain...")
print("=" * 50)
try:
    result = decision_chain.invoke({"input": test_input})
    print("Success!")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
