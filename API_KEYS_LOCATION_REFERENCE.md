# API Keys - Location Reference Map

Quick lookup for where each API key is used in the codebase.

---

## 🔑 NVIDIA_API_KEY
**Purpose:** LLM generation for NVIDIA NIM

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| Agent Decision | `config.py` | 22 | LLM for routing decisions |
| Conversation | `config.py` | 26 | LLM for general chat |
| Web Search | `config.py` | 30 | LLM for search results processing |
| RAG - Main LLM | `config.py` | 49 | Core RAG response generation |
| RAG - Summarizer | `config.py` | 50 | Image summarization |
| RAG - Chunker | `config.py` | 51 | Document chunking |
| RAG - Response Gen | `config.py` | 52 | Final response generation |
| RAG - Embeddings | `config.py` | 48 | Local FastEmbed embeddings |
| Medical CV | `config.py` | 76 | Medical image analysis |

**Environment Variable:** `NVIDIA_API_KEY`
**Required:** ✅ Yes
**Service:** [NVIDIA NIM](https://build.nvidia.com/)

---

## 🎙️ ELEVEN_LABS_API_KEY
**Purpose:** Text-to-Speech API for audio synthesis

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| ElevenLabs Init | `app.py` | 50 | Initialize ElevenLabs client |
| Speech Config | `config.py` | 83 | Store API key reference |
| Speech Request Handler | `app.py` | 345 | Pass key in request headers |

**Environment Variable:** `ELEVEN_LABS_API_KEY`
**Required:** ✅ Yes
**Service:** [ElevenLabs](https://elevenlabs.io/)
**Usage Pattern:**
```python
client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)
```

---

## 🔍 TAVILY_API_KEY
**Purpose:** Real-time medical web search

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| Config Storage | `config.py` | 126 | Store API key reference |
| Tavily Search | `agents/web_search_processor_agent/tavily_search.py` | 21-22 | Web search execution and validation |

**Environment Variable:** `TAVILY_API_KEY`
**Required:** ✅ Yes
**Service:** [Tavily API](https://tavily.com/)
**Usage Pattern:**
```python
if not os.getenv("TAVILY_API_KEY"):
    return "Web search is currently unavailable"
```

---

## 🤗 HUGGINGFACE_TOKEN
**Purpose:** Access to reranker models for document relevance scoring

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| RAG Config | `config.py` | 59 | Store token for HF model access |
| Reranker Component | `agents/rag_agent/reranker.py` | N/A | Used by cross-encoder model |

**Environment Variable:** `HUGGINGFACE_TOKEN`
**Required:** ✅ Yes
**Service:** [HuggingFace](https://huggingface.co/)
**Model Used:** `cross-encoder/ms-marco-TinyBERT-L-6`

---

## 🗄️ QDRANT_URL (Optional)
**Purpose:** Cloud-hosted vector database endpoint

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| RAG Config | `config.py` | 44 | Vector store connection |

**Environment Variable:** `QDRANT_URL`
**Required:** ❌ No (local mode is default)
**Service:** [Qdrant Cloud](https://cloud.qdrant.io/)
**Default Behavior:**
```python
self.url = os.getenv("QDRANT_URL")  # None if not set = local mode
```

---

## 🔐 QDRANT_API_KEY (Optional)
**Purpose:** Authentication for cloud Qdrant vector database

| Component | File | Lines | Usage |
|-----------|------|-------|-------|
| RAG Config | `config.py` | 45 | Vector store authentication |

**Environment Variable:** `QDRANT_API_KEY`
**Required:** ❌ No (only for cloud mode)
**Service:** [Qdrant Cloud](https://cloud.qdrant.io/)
**Note:** Only needed when `QDRANT_URL` is configured

---

## 📋 Configuration Loading Flow

```
.env file
   ↓
load_dotenv() in config.py (line 19)
   ↓
os.getenv("KEY_NAME") in respective Config classes
   ↓
Stored in config object instances
   ↓
Used throughout application
```

---

## 🔧 Adding/Modifying API Keys

### To add a new API key:

1. **Add to .env file:**
   ```env
   NEW_API_KEY=your_value_here
   ```

2. **Load in config.py:**
   ```python
   self.new_api_key = os.getenv("NEW_API_KEY")
   ```

3. **Use in your code:**
   ```python
   config.your_config_section.new_api_key
   ```

### To verify an API key is loaded:

```bash
# Check if key exists in environment
grep "API_KEY" .env

# Verify in Python
python -c "from config import Config; c = Config(); print(c.your_section.api_key)"
```

---

## 🚨 Missing/Invalid Keys - Effects

| Key | Missing | Invalid | Effect |
|-----|---------|---------|--------|
| NVIDIA_API_KEY | 🔴 Critical | 🔴 Critical | Application won't start - no LLM |
| ELEVEN_LABS_API_KEY | 🟠 Important | 🟠 Important | Speech synthesis fails |
| TAVILY_API_KEY | 🟠 Important | 🟡 Warning | Web search unavailable (logs message) |
| HUGGINGFACE_TOKEN | 🟠 Important | 🟠 Important | Reranking fails |
| QDRANT_* | 🟢 Optional | 🟢 Optional | Falls back to local database |

---

## 📊 Usage Statistics (Approximate)

Based on typical deployment:

| Service | Estimated Daily Calls | Free Tier Limit | Cost per Million |
|---------|----------------------|-----------------|------------------|
| NVIDIA NIM | 500-2000 | Depends on deployment | Variable |
| ElevenLabs | 50-500 | 10,000 chars | $0.30 |
| Tavily | 10-100 | 100 calls | $2-5 |
| HuggingFace | 100-1000 | Unlimited | Free |
| Qdrant | 500-2000 | Local free | $25-200 |

*Note: Actual usage varies based on deployment and user activity*

---

## 🔄 Key Rotation / Update

To rotate an API key:

1. Generate new key in service dashboard
2. Update `.env` file: `KEY_NAME=new_value`
3. Restart application
4. Revoke old key in service dashboard

No code changes needed - configuration is read at startup.

---

## ✅ Validation Commands

```bash
# Check all keys are set
echo "Checking API keys..."
grep -E "NVIDIA_API_KEY|ELEVEN_LABS|TAVILY|HUGGINGFACE" .env

# Test individual keys (examples)
# Test Google API (requires curl + jq)
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
   -H "Authorization: Bearer $NVIDIA_API_KEY" \
   -H "Content-Type: application/json" \
   -d '{"model":"ibm/granite-3.0-8b-instruct","messages":[{"role":"user","content":"Hello"}]}'

# Validate .env syntax
python -c "from dotenv import load_dotenv; load_dotenv(); print('✓ .env loaded successfully')"
```

---

## 📞 Support & Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not found" | Check .env file syntax and .env file is in project root |
| "401 Unauthorized" | Key is invalid - regenerate from service |
| "403 Forbidden" | Check API permissions in service dashboard |
| "429 Too Many Requests" | Rate limit hit - check quota or upgrade plan |
| Keys not loading | Ensure `from dotenv import load_dotenv` runs before Config init |

---

Last Updated: 12 May 2026
