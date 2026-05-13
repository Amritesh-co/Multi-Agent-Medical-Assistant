# Multi-Agent Medical Assistant - API Keys Setup Guide

## Overview
This document lists all API keys required to run the Multi-Agent Medical Assistant. The project has been optimized to use **NVIDIA NIM** for LLM generation, with local embeddings for RAG.

---

## 📋 Required API Keys

### 1. **NVIDIA_API_KEY** ⭐ (Primary LLM)
- **Location:** `.env` file
- **Environment Variable Name:** `NVIDIA_API_KEY`
- **Used By:**
  - Agent Decision System (routing queries to appropriate agents)
  - Conversation Agent (general chat and medical discussions)
  - RAG Agent (retrieval-augmented generation with embeddings)
  - Web Search Agent (processing web search results)
  - Medical CV Config (image analysis assistance)
  - All LLM operations throughout the application

- **Models Used:**
  - **LLM Model:** `ibm/granite-3.0-8b-instruct` (all conversational tasks)
  - **Embedding Model:** local `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5` by default)

- **How to Get It:**
  1. Go to your NVIDIA NIM / Build portal or hosted NIM deployment
  2. Generate an OpenAI-compatible API key
  3. Copy the key to your `.env` file

- **Configuration in Code:**
  - `config.py` - Lines 22, 26, 30, 49-53, 76
  - Uses `ChatOpenAI` with `base_url=https://integrate.api.nvidia.com/v1`

---

### 2. **ELEVEN_LABS_API_KEY** (Text-to-Speech)
- **Location:** `.env` file
- **Environment Variable Name:** `ELEVEN_LABS_API_KEY`
- **Used By:**
  - Speech synthesis for converting text responses to audio
  - ElevenLabs client in `app.py` (line 50)
  - `config.py` SpeechConfig class (line 83)

- **Model Used:**
  - Default Voice ID: `21m00Tcm4TlvDq8ikWAM` (Rachel voice)

- **How to Get It:**
  1. Sign up at [ElevenLabs](https://elevenlabs.io/)
  2. Go to your dashboard → API Keys
  3. Create or copy your API key
  4. Add to `.env` file
  5. Free tier includes 10,000 characters/month

- **Configuration in Code:**
  - `config.py` - SpeechConfig class (line 83)
  - `app.py` - Lines 50, 345

---

### 3. **TAVILY_API_KEY** (Web Search)
- **Location:** `.env` file
- **Environment Variable Name:** `TAVILY_API_KEY`
- **Used By:**
  - Real-time web search for current medical information
  - Web search processor agent for retrieving up-to-date medical data
  - Falls back gracefully if not configured

- **How to Get It:**
  1. Sign up at [Tavily](https://tavily.com/)
  2. Go to your account dashboard
  3. Generate API key
  4. Add to `.env` file
  5. Free tier available

- **Configuration in Code:**
  - `config.py` - Line 126
  - `agents/web_search_processor_agent/tavily_search.py` - Line 21-22 (validates if key is set)

---

### 4. **HUGGINGFACE_TOKEN** (Model Reranking)
- **Location:** `.env` file
- **Environment Variable Name:** `HUGGINGFACE_TOKEN`
- **Used By:**
  - Reranker model for improving RAG retrieval quality
  - Cross-encoder model: `cross-encoder/ms-marco-TinyBERT-L-6`
  - Improves relevance of retrieved medical documents

- **How to Get It:**
  1. Sign up at [Hugging Face](https://huggingface.co/)
  2. Go to Settings → Access Tokens
  3. Create a new token with "read" access
  4. Add to `.env` file

- **Configuration in Code:**
  - `config.py` - RAGConfig class (line 59)
  - Used by Reranker component for relevance scoring

---

### 5. **QDRANT_URL** (Optional - Vector Database)
- **Location:** `.env` file
- **Environment Variable Name:** `QDRANT_URL`
- **Used By:**
  - Cloud-hosted Qdrant vector database (optional)
  - If not set, uses local Qdrant database at `./data/qdrant_db`

- **How to Get It:**
  1. Set up Qdrant cloud at [Qdrant Cloud](https://cloud.qdrant.io/)
  2. Create a cluster
  3. Copy the cluster URL
  4. Add to `.env` file

- **Configuration in Code:**
  - `config.py` - RAGConfig class (line 44)
  - Defaults to local mode if not provided

---

### 6. **QDRANT_API_KEY** (Optional - Vector Database Auth)
- **Location:** `.env` file
- **Environment Variable Name:** `QDRANT_API_KEY`
- **Used By:**
  - Authentication for cloud-hosted Qdrant database
  - Only needed if using `QDRANT_URL`
  - Local mode doesn't require this

- **How to Get It:**
  1. Log in to Qdrant Cloud
  2. Go to your cluster settings
  3. Copy the API key
  4. Add to `.env` file

- **Configuration in Code:**
  - `config.py` - RAGConfig class (line 45)
  - Paired with QDRANT_URL

---

## 📝 Complete .env Template

Copy this to your `.env` file and fill in the values:

```env
# REQUIRED: NVIDIA NIM API Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=z-ai/glm-5.1
NVIDIA_MAX_TOKENS=8192

# REQUIRED: ElevenLabs Text-to-Speech API
ELEVEN_LABS_API_KEY=your_eleven_labs_key_here

# REQUIRED: Tavily Web Search API
TAVILY_API_KEY=your_tavily_api_key_here

# REQUIRED: HuggingFace Token for Reranking
HUGGINGFACE_TOKEN=your_huggingface_token_here

# OPTIONAL: Qdrant Cloud Database (leave empty for local)
QDRANT_URL=
QDRANT_API_KEY=
```

---

## 🔄 Summary Table

| API Key | Required | Purpose | Service | Free Tier |
|---------|----------|---------|---------|-----------|
| NVIDIA_API_KEY | ✅ Yes | LLM generation | NVIDIA NIM | Depends on plan |
| ELEVEN_LABS_API_KEY | ✅ Yes | Text-to-Speech | ElevenLabs | Yes (10k chars/mo) |
| TAVILY_API_KEY | ✅ Yes | Web Search | Tavily | Yes |
| HUGGINGFACE_TOKEN | ✅ Yes | Model Reranking | HuggingFace | Yes |
| QDRANT_URL | ❌ No | Vector DB (Cloud) | Qdrant | Optional |
| QDRANT_API_KEY | ❌ No | Vector DB Auth | Qdrant | Optional |

---

## 🚀 Getting Started

1. **Get all 4 required API keys** from the services listed above
2. **Create/update `.env` file** in the project root with all keys
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Run the application:** `python app.py` or use Docker
5. **Access the UI:** Navigate to `http://localhost:8001`

---

## ✨ Key Improvements with NVIDIA NIM Optimization

- ✅ Replaced Gemini chat calls with **NVIDIA NIM** OpenAI-compatible API
- ✅ Using **local FastEmbed embeddings** for RAG without extra API keys
- ✅ Improved flexibility by centralizing model selection in `.env`
- ✅ Faster inference with cloud-hosted NIM model access
- ✅ Better support for streaming/thinking models like `z-ai/glm-5.1`
- ✅ Scalable API-based solution (no local LLM management needed)

---

## 🔧 Model Configuration

All models are configured in `config.py`:

```python
# Primary LLM for all tasks
ChatOpenAI(model="ibm/granite-3.0-8b-instruct", base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_API_KEY"), temperature=0.X)

# Embeddings for semantic search
FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
```

Temperatures are tuned per use case:
- **Agent Decision:** 0.1 (strict routing)
- **Conversation:** 0.7 (natural dialogue)
- **Web Search:** 0.3 (balanced information retrieval)
- **RAG:** 0.3 (factual responses)
- **Summarization:** 0.5 (balanced summaries)
- **Chunking:** 0.0 (deterministic)

---

## ❓ Troubleshooting

### Missing NVIDIA_API_KEY
```
Error: API key not found for 'NVIDIA_API_KEY'
```
**Solution:** Add `NVIDIA_API_KEY=your_key` to `.env` file

### Rate Limits
If you hit rate limits on Gemini API:
- Check your [Google AI quota](https://console.cloud.google.com/)
- Upgrade to a paid tier if needed
- Implement request throttling

### Vector Database Errors
If using cloud Qdrant:
- Verify `QDRANT_URL` format is correct
- Ensure `QDRANT_API_KEY` is valid
- Check network connectivity to the URL

---

## 📚 Additional Resources

- [Google Gemini API Docs](https://ai.google.dev/docs)
- [ElevenLabs Documentation](https://elevenlabs.io/docs)
- [Tavily Search API](https://tavily.com/docs)
- [HuggingFace Guide](https://huggingface.co/docs)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
- [LangChain Google Integration](https://python.langchain.com/docs/integrations/llms/google_generative_ai)
