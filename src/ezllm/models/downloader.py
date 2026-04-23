try:
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError:
    hf_hub_download = None


def download_model_artifact(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    local_dir: str | None = None,
    repo_type: str = "model",
) -> str:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required for `models download`.")

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if local_dir is not None:
        kwargs["local_dir"] = local_dir

    return str(hf_hub_download(**kwargs))
