# MeetMind - Advanced RAG-Powered Knowledge Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system that demonstrates advanced ML/AI engineering practices, vector search, and modern software architecture.

## Features

- **Advanced RAG Pipeline**: Custom-built retrieval and generation system with multiple embedding models
- **Multi-Modal Support**: Text, document, and audio transcription capabilities
- **Vector Search**: High-performance similarity search using Qdrant vector database
- **Production-Ready**: Docker containerization, CI/CD, monitoring, and comprehensive testing
- **Scalable Architecture**: Modular design with optional LangChain/LlamaIndex integration
- **Real-time Processing**: FastAPI-based API with async processing capabilities

## Tech Stack

- **Language**: Python 3.10+
- **Web API**: FastAPI + Uvicorn
- **Embeddings**: sentence-transformers, OpenAI embeddings
- **Vector DB**: Qdrant (Docker), with Pinecone/Milvus alternatives
- **RAG Orchestration**: Custom implementation + optional LangChain/LlamaIndex
- **Storage**: SQLite (dev), PostgreSQL (prod)
- **Transcription**: faster-whisper, OpenAI Whisper API
- **Testing**: pytest with comprehensive coverage
- **Container**: Docker & docker-compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Sentry, Prometheus/Grafana
- **Code Quality**: black, isort, flake8, mypy

## Architecture

```
meet-mind/
├── app/                    # Core application
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Configuration and core logic
│   ├── models/           # Data models and schemas
│   ├── services/         # Business logic services
│   ├── rag/              # RAG pipeline components
│   └── utils/            # Utility functions
├── tests/                 # Comprehensive test suite
├── docker/                # Docker configurations
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── monitoring/            # Monitoring and observability
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd meet-mind
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start services with Docker**
   ```bash
   docker-compose up -d
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## API Endpoints

### Core RAG Operations
- `POST /api/v1/rag/query` - Query the knowledge base
- `POST /api/v1/rag/ingest` - Ingest new documents
- `GET /api/v1/rag/sources` - Get source documents
- `POST /api/v1/rag/chat` - Interactive chat with context

### Document Management
- `POST /api/v1/documents/upload` - Upload documents
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete document

### Vector Operations
- `POST /api/v1/vectors/search` - Vector similarity search
- `GET /api/v1/vectors/stats` - Vector database statistics

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_rag/
pytest tests/test_api/
```

## Docker

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Monitoring

- **Health Checks**: Built-in health monitoring endpoints
- **Metrics**: Prometheus metrics collection
- **Logging**: Structured logging with correlation IDs
- **Error Tracking**: Sentry integration for production

## Development

### Code Quality
```bash
# Format code
black app/ tests/
isort app/ tests/

# Lint code
flake8 app/ tests/
mypy app/

# Run all quality checks
make quality
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Deployment

### Production
```bash
# Build production image
docker build -t meet-mind:latest .

# Run with production settings
docker run -p 8000:8000 meet-mind:latest
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=openai_key
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_ENABLED=true
```
