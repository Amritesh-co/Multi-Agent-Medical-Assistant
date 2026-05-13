"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings

# Load environment variables from .env file
load_dotenv()


def build_nim_chat_model(temperature: float):
    return ChatOpenAI(
        model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        openai_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        max_tokens=int(os.getenv("NVIDIA_MAX_TOKENS", "8192")),
    )


def build_nim_vision_model(temperature: float):
    """Build a vision-capable model for multimodal image analysis."""
    return ChatOpenAI(
        model=os.getenv("NVIDIA_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct"),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        openai_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        max_tokens=int(os.getenv("NVIDIA_MAX_TOKENS", "8192")),
    )


def build_embedding_model():
    return FastEmbedEmbeddings(model_name=os.getenv("NVIDIA_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = build_nim_chat_model(temperature=0.1)

class ConversationConfig:
    def __init__(self):
        self.llm = build_nim_chat_model(temperature=0.7)

class WebSearchConfig:
    def __init__(self):
        self.llm = build_nim_chat_model(temperature=0.3)
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        # Use local FastEmbed embeddings so chat generation can move fully to NVIDIA NIM.
        self.embedding_model = build_embedding_model()
        self.llm = build_nim_chat_model(temperature=0.3)
        self.summarizer_model = build_nim_chat_model(temperature=0.5)
        self.chunker_model = build_nim_chat_model(temperature=0.0)
        self.response_generator_model = build_nim_chat_model(temperature=0.3)
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = build_nim_chat_model(temperature=0.1)
        self.vision_llm = build_nim_vision_model(temperature=0.1)  # Vision model for image analysis

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "XrExE9yKIg1WjnnlVkGX"    # Matilda (Knowledgeable, Professional)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8001
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history

# # Example usage
# config = Config()