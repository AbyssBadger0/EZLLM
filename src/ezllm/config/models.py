from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    host: str = "127.0.0.1"
    proxy_port: int = 8888
    llama_port: int = 8889
    log_dir: str
    state_dir: str


class LlamaConfig(BaseModel):
    server_bin: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    mmproj_path: str | None = None
    ctx_size: int = 32768
    n_predict: int = 4096
    parallel: int = 1
    gpu_layers: int = 999
    batch_size: int = 512
    flash_attn: str = "auto"


class Settings(BaseModel):
    runtime: RuntimeConfig
    llama: LlamaConfig
