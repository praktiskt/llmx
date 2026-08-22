import os
import random
import re
import string
from datetime import date


class Config:
    CONTENT_THRESHOLD = 5000
    MAX_TOOL_RESULT_CHARS = 8000
    GREP_MAX_MATCHES = 50
    LLM_TIMEOUT = 60
    FETCH_TIMEOUT = 30
    SEARCH_TIMEOUT = 10
    HEAD_TIMEOUT = 5

    @staticmethod
    def response_format():
        return os.environ.get("LLM_RESPONSE_FORMAT", None)

    @staticmethod
    def is_stream():
        if Config.response_format() is not None:
            return False
        return os.environ.get("LLM_STREAM", "True").lower() == "true"

    @staticmethod
    def tools_enabled():
        return os.environ.get("LLM_ENABLE_TOOLS", "True").lower() == "true"

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
    def get_system_prompt() -> str:
        today = date.today().isoformat()
        return (
            f"Today is: {today}. "
            "When making multiple independent tool calls, batch them in a single response for efficiency. "
            "IMPORTANT: Use fetch() to get content. If content is large, it returns a file_id (6 lowercase alphanumeric chars). "
            "When you need to read, grep, or summarize multiple files, use file_ids=[...] in a single call for efficiency. "
            "Only use read_file, grep_file, or summarize with that exact file_id returned by fetch. "
            "NEVER invent or guess file_ids - only use IDs explicitly returned by fetch."
        )

    @staticmethod
    def generate_file_id() -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    @staticmethod
    def validate_file_id(file_id: str) -> str | None:
        if not file_id:
            return "file_id is required"
        if not re.match(r"^[a-z0-9]{6}$", file_id):
            return f"invalid file_id '{file_id}'. Must be 6 lowercase alphanumeric characters (e.g., 'abc123'). Do NOT invent file_ids - only use IDs returned by fetch."
        return None
