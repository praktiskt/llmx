import os
import random
import re
import string
from datetime import date


class Config:
    MAX_TOOL_RESULT_CHARS = int(os.environ.get("LLM_MAX_TOOL_RESULT_CHARS", "16000"))
    GREP_MAX_MATCHES = int(os.environ.get("LLM_GREP_MAX_MATCHES", "100"))
    LLM_TIMEOUT = 60
    FETCH_TIMEOUT = 30
    SEARCH_TIMEOUT = 10
    MAX_CONTEXT_CHARS = int(os.environ.get("LLM_MAX_CONTEXT_CHARS", "200000"))
    LOCAL_MAX_FILES = int(os.environ.get("LLM_LOCAL_MAX_FILES", "300"))
    LOCAL_MAX_FILE_BYTES = int(os.environ.get("LLM_LOCAL_MAX_FILE_BYTES", "5000000"))

    @staticmethod
    def response_format():
        return os.environ.get("LLM_RESPONSE_FORMAT", None)

    @staticmethod
    def is_stream():
        if Config.response_format() is not None:
            return False
        return os.environ.get("LLM_STREAM", "True").lower() == "true"

    @staticmethod
    def allowed_tools() -> set[str] | None:
        """Allowlist from LLM_TOOLS (comma-separated names).

        Unset -> all tools. Set -> exactly those tools; empty value disables all.
        """
        raw = os.environ.get("LLM_TOOLS")
        if raw is None:
            return None
        return {name.strip() for name in raw.split(",") if name.strip()}

    @staticmethod
    def color_output_enabled():
        if os.environ.get("NO_COLOR"):
            return False
        return os.environ.get("LLM_DISABLE_COLOR_OUTPUT", "False").lower() != "true"

    @staticmethod
    def thinking_enabled():
        return os.environ.get("LLM_SHOW_THINKING", "True").lower() == "true"

    @staticmethod
    def markdown_fetch_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_FETCH_PROXY")

    @staticmethod
    def markdown_search_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_SEARCH_PROXY")

    @staticmethod
    def markdown_image_search_proxy() -> str | None:
        return os.environ.get("LLM_MARKDOWN_IMAGE_SEARCH_PROXY")

    @staticmethod
    def fetch_allow_private() -> bool:
        return os.environ.get("LLM_FETCH_ALLOW_PRIVATE", "False").lower() == "true"

    @staticmethod
    def local_files_enabled() -> bool:
        return os.environ.get("LLM_LOCAL_FILES", "False").lower() == "true"

    @staticmethod
    def interactive_enabled() -> bool:
        return os.environ.get("LLM_INTERACTIVE", "False").lower() == "true"

    @staticmethod
    def mcp_servers() -> dict | None:
        raw = os.environ.get("LLM_MCP_SERVERS")
        if not raw:
            return None
        import json as _json

        try:
            data = _json.loads(raw)
        except Exception as e:
            import sys as _sys

            print(f"[mcp] invalid LLM_MCP_SERVERS JSON: {e}", file=_sys.stderr)
            return None
        if not isinstance(data, dict):
            import sys as _sys

            print("[mcp] LLM_MCP_SERVERS must be a JSON object", file=_sys.stderr)
            return None
        return data

    @staticmethod
    def mcp_timeout() -> float:
        try:
            return float(os.environ.get("LLM_MCP_TIMEOUT", "30"))
        except ValueError:
            return 30.0

    @staticmethod
    def get_system_prompt() -> str:
        today = date.today().isoformat()
        prompt = (
            f"Today is: {today}. "
            "When making multiple independent tool calls, batch them in a single response for efficiency. "
            "IMPORTANT: fetch() and search() store content as memory://<id> (6 lowercase alphanumeric chars). "
            "When you need to read, grep, or summarize multiple documents, use sources=[...] in a single call for efficiency. "
            "Only address documents by their exact memory://<id> returned by fetch or search. "
            "NEVER invent or guess memory ids."
        )
        if Config.local_files_enabled():
            prompt += (
                " sources=[...] also accepts local filesystem paths/globs relative to the "
                "current working directory (e.g. 'src/**/*.py'); list_files lists local files "
                "by glob/regex. Local access is restricted: no absolute paths, no '..', nothing "
                "resolving outside the current directory."
            )
        mcp = Config.mcp_servers()
        if mcp:
            prompt += f" MCP servers available: {', '.join(sorted(mcp.keys()))} (tools as <server>__<tool>)."
        return prompt

    @staticmethod
    def generate_file_id() -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    @staticmethod
    def validate_file_id(file_id: str) -> str | None:
        if not file_id:
            return "file_id is required"
        if not re.match(r"^[a-z0-9]{6}$", file_id):
            return f"invalid id '{file_id}' - must be 6 lowercase alphanumeric characters; use the full memory://<id> exactly as returned by fetch or search."
        return None
