import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = DATA_DIR / "storage"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RAGConfig:
    # Storage Paths
    base_dir: Path = BASE_DIR
    storage_dir: Path = STORAGE_DIR
    db_path: Path = STORAGE_DIR / "documents.db"
    vector_index_path: Path = STORAGE_DIR / "vector_index.npz"
    metadata_store_path: Path = STORAGE_DIR / "vector_metadata.json"

    # Ingestion & Chunking
    chunk_size: int = int(os.getenv("DEFAULT_CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 80))

    # Retrieval Configuration
    top_k: int = int(os.getenv("DEFAULT_TOP_K", 4))
    retrieval_mode: str = os.getenv("DEFAULT_RETRIEVAL_MODE", "hybrid")  # "hybrid", "dense", "bm25"
    hybrid_alpha: float = 0.6  # 0.6 dense + 0.4 sparse weight
    min_similarity_threshold: float = 0.25

    # LLM & Embedding Settings
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")  # "local", "openai", "gemini"
    llm_provider: str = os.getenv("LLM_PROVIDER", "offline")  # "offline", "openai", "gemini", "ollama"
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "extractive-offline-v1")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # MLflow Settings
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "Enterprise-Document-RAG")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(STORAGE_DIR / 'mlflow.db').as_posix()}")
    enable_mlflow: bool = True
