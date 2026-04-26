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
    ctx_size: int = 200000
    n_predict: int = 81920
    parallel: int = 1
    gpu_layers: int = 999
    batch_size: int = 512
    flash_attn: str = "on"
    cache_k_type: str = "q8_0"
    cache_v_type: str = "q8_0"
    temp: str = "0.7"
    top_p: str = "0.95"
    top_k: str = "20"
    reasoning: str = "auto"
    reasoning_format: str = "deepseek"
    reasoning_budget: str = "-1"


class Settings(BaseModel):
    runtime: RuntimeConfig
    llama: LlamaConfig
