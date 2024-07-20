from pathlib import Path


repo_root = Path(__file__).parent

raise RuntimeError(
    f"Python is looking for PyTensor in {repo_root}, but it's "
    f"actually located in {repo_root / 'pytensor'}. Probably "
    f"you need to change your working directory from {Path.cwd()} "
    f"to {repo_root}."
)
