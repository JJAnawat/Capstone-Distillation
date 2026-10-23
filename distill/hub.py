"""Back up a run to a private Hugging Face dataset repo: a safety net for data that otherwise lives only on the VM.
Files keep their local paths, so `hf download REPO --repo-type dataset --local-dir .` puts them back where every
stage resumes from them. Unchanged files are not uploaded again."""
import os
from pathlib import Path

from distill.data import PREPARED_DIR

RUN_FILES = ["*.jsonl", "*.json"]                 # generations (with the raw thinking), kept, judge, final, run logs
RATINGS = ["*.complexity.jsonl", "*.runs.jsonl"]  # the prompt caches next to them are rebuilt by `prepare`


def push(cfg, out, repo=None, api=None):
    """Upload the run directory `out` and the complexity ratings to `repo` (default: config hub.repo). Returns False
    when no repo is set. Only a private repo is accepted: the data is unreviewed and No Robots outputs are
    non-commercial."""
    repo = repo or (cfg.get("hub") or {}).get("repo")
    if not repo:
        print("[push] no hub.repo in the config; skipping the backup")
        return False
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise RuntimeError(f"{repo} is public; the backup only goes to a private repo")
    for folder, patterns in ((Path(out), RUN_FILES), (PREPARED_DIR, RATINGS)):
        path = Path(os.path.relpath(folder)).as_posix()
        if path.startswith(".."):
            raise ValueError(f"{folder} is outside the working directory")
        if any(any(folder.glob(p)) for p in patterns):
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, path_in_repo=path,
                              allow_patterns=patterns, commit_message=f"backup {path}")
    print(f"[push] {out} and the ratings are backed up to https://huggingface.co/datasets/{repo}")
    return True
