from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR=Path(__file__).resolve().parents[1]  # the backend/folder

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    #------------Models-----------
    llm_provider: str = "ollama" # "ollama" or "groq"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "embeddinggemma:300m"
    ollama_chat_model: str = "qwen2.5:3b"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # -----Chunking and retrieval--------
    chunk_size: int=1000
    chunk_overlap: int=150
    top_k: int=5  # max sources sent to the LLM
    max_distance: float=0.65   # cosine distance cutoff(tune with /api/search)

    #--------Uploads and server---------
    max_upload_mb: int=25
    cors_origins: list[str]=['http://localhost:5173']  #vite dev server

    #--------Paths-----------
    @property
    def data_dir(self)-> Path:
        return BASE_DIR / "data"

    @property
    def upload_dir(self)-> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self)-> Path:
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self)-> Path:
        return self.data_dir / "askdocs.db"

    def ensure_dirs(self)-> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

settings=Settings()