import re
from pathlib import Path

from .cache import Cache
from .config import Config

_GLOB_MAGIC = re.compile(r"[*?\[\]]")


def _label(path: Path, cwd: Path) -> str:
    try:
        return path.relative_to(cwd).as_posix()
    except ValueError:
        return path.as_posix()


def _validated(candidate: Path, root: Path) -> Path | None:
    """Resolved path if a file under root, else None."""
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


def resolve(paths: list[str]) -> tuple[list[Path], list[str]]:
    """Expand relative paths/globs under cwd into validated file lists.

    Returns (files, errors). Rejects absolute paths, '..' components, and
    anything resolving outside cwd (including symlinks pointing out).
    """
    files: list[Path] = []
    seen: set[Path] = set()
    errors: list[str] = []
    cwd = Path.cwd()
    root = cwd.resolve()

    def add(resolved: Path, candidate: Path) -> None:
        if resolved not in seen:
            seen.add(resolved)
            files.append(candidate)

    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            errors.append("Error: empty path provided")
            continue

        rel = Path(raw)
        if rel.is_absolute():
            errors.append(
                f"Error: path '{raw}' is absolute - use paths relative to the current directory"
            )
            continue
        if ".." in rel.parts:
            errors.append(
                f"Error: path '{raw}' must not contain '..' - only paths under "
                "the current directory are allowed"
            )
            continue

        if _GLOB_MAGIC.search(raw):
            try:
                matches = sorted(cwd.glob(raw))
            except (OSError, ValueError) as e:
                errors.append(f"Error: invalid glob '{raw}': {e}")
                continue
            matched = 0
            truncated = False
            for match in matches:
                resolved = _validated(match, root)
                if resolved is None:
                    continue
                add(resolved, match)
                matched += 1
                if len(files) >= Config.LOCAL_MAX_FILES:
                    truncated = True
                    break
            if matched == 0 and not truncated:
                errors.append(f"Error: path '{raw}' matched no files")
            elif truncated:
                errors.append(
                    f"Error: stopped after {Config.LOCAL_MAX_FILES} files - narrow the glob '{raw}'"
                )
            continue

        candidate = cwd / rel
        if candidate.is_dir():
            errors.append(
                f"Error: path '{raw}' is a directory - use a glob like "
                f"'{Path(raw).as_posix().rstrip('/')}/**' or a specific file"
            )
            continue
        resolved = _validated(candidate, root)
        if resolved is None:
            errors.append(f"Error: file {raw} not found")
            continue
        add(resolved, candidate)

    return files, errors


def _load(file: Path) -> str:
    """Read a validated file as text; returns 'Error: ...' string on failure."""
    try:
        data = file.read_bytes()
    except OSError as e:
        return f"Error: cannot read file: {e}"
    if b"\0" in data[:8192]:
        return "Error: binary file - cannot display"
    if len(data) > Config.LOCAL_MAX_FILE_BYTES:
        return f"Error: file too large ({len(data)} bytes, max {Config.LOCAL_MAX_FILE_BYTES})"
    return data.decode("utf-8", errors="replace")


def read_entries(
    paths: list[str], offset: int | None = None, limit: int | None = None
) -> list[str]:
    """Per-path read results formatted identically to Cache.read entries."""
    files, errors = resolve(paths)
    results = list(errors)
    cwd = Path.cwd()
    for file in files:
        content = _load(file)
        if content.startswith("Error:"):
            results.append(content)
            continue
        results.append(Cache._format_read(_label(file, cwd), content, offset, limit))
    return results


def grep_entries(
    paths: list[str],
    pattern: str,
    is_regex: bool = False,
    ignore_case: bool = False,
    context: int = 0,
) -> list[str]:
    """Per-path grep results formatted identically to Cache.grep entries."""
    files, errors = resolve(paths)
    results = list(errors)
    cwd = Path.cwd()
    for file in files:
        content = _load(file)
        if content.startswith("Error:"):
            results.append(content)
            continue
        results.append(
            Cache._grep_content(
                _label(file, cwd), content, pattern, is_regex, ignore_case, context
            )
        )
    return results


def contents(paths: list[str]) -> list[tuple[str | None, str]]:
    """(label, text) pairs; label None means text is an error message."""
    files, errors = resolve(paths)
    pairs: list[tuple[str | None, str]] = [(None, e) for e in errors]
    cwd = Path.cwd()
    for file in files:
        text = _load(file)
        label = None if text.startswith("Error:") else _label(file, cwd)
        pairs.append((label, text))
    return pairs


def list_entries(
    pattern: str = "**/*", regex: str | None = None, ignore_case: bool = False
) -> str:
    """List files under cwd matching an optional glob and/or path regex."""
    cwd = Path.cwd()
    root = cwd.resolve()

    if regex:
        try:
            rx = re.compile(regex, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return f"Error: invalid regex: {e}"
    else:
        rx = None

    if not isinstance(pattern, str) or not pattern.strip():
        pattern = "**/*"
    if ".." in Path(pattern).parts or Path(pattern).is_absolute():
        return (
            "Error: pattern must stay under the current directory - "
            "no absolute paths or '..' components"
        )

    try:
        matches = sorted(cwd.glob(pattern))
    except (OSError, ValueError) as e:
        return f"Error: invalid glob '{pattern}': {e}"

    names: list[str] = []
    truncated = False
    for match in matches:
        if _validated(match, root) is None:
            continue
        rel = _label(match, cwd)
        if rx is not None and not rx.search(rel):
            continue
        names.append(rel)
        if len(names) >= Config.LOCAL_MAX_FILES:
            truncated = True
            break

    header = f"{len(names)} file(s)"
    if truncated:
        header += f" (stopped at {Config.LOCAL_MAX_FILES} - narrow the pattern)"
    body = "\n".join(names) if names else "no matching files"
    return f"{header}\n{body}"
