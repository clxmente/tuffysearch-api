# TuffySearch API 🎓

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.24-FF6B6B)](https://www.langchain.com/)

A powerful semantic search API for the California State University, Fullerton course catalog. TuffySearch enables intelligent course discovery through natural language queries, making it easier for students to find relevant courses based on their interests and requirements. Built with FastAPI and powered by HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model for embeddings, the API uses FAISS (via LangChain) as an in-memory vector database for lightning-fast similarity search across the entire course catalog.

## ✨ Features

- 🔍 **Semantic Search**: Find courses using natural language queries
- 🎯 **Course Level Filtering**: Filter results by course level (100-599)
- ⚡ **Rate Limiting**: 100 requests per hour per IP
- 🚀 **Fast Performance**: Built with FastAPI and FAISS for efficient vector search
- 🤖 **Advanced Embeddings**: Powered by HuggingFace's sentence-transformers

## 🛠️ Technologies

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Library for efficient similarity search
- **HuggingFace Embeddings**: State-of-the-art sentence embeddings
- **Docker**: Containerization for easy deployment

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tuffysearch-api.git
cd tuffysearch-api
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Run the setup script to create embeddings:
```bash
uv run setup_embeddings.py
```

4. Start the API server:
```bash
uv run fastapi dev main.py
```

### Docker Deployment

Build and run using Docker:
```bash
docker build -t tuffysearch-api .
docker run -p 80:80 tuffysearch-api --restart=always
# OR with compose
docker compose up -d --build
```

## 📚 API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Request

```bash
curl -X GET "http://localhost:8000/search?q=income%20inequality&min_level=100&max_level=400"
```

## 📝 License

This project is licensed under the MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
