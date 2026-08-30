import sys

from .config import Config


class Log:
    @staticmethod
    def stdout(msg, **kwargs) -> None:
        print(msg, file=sys.stdout, **kwargs)

    @staticmethod
    def stderr(msg, **kwargs) -> None:
        print(msg, file=sys.stderr, **kwargs)


class Color:
    RESET = "\033[0m"
    TOOL_COLORS = {
        "fetch": "\033[2;36m",
        "search": "\033[2;32m",
        "read": "\033[2;34m",
        "read_file": "\033[2;34m",
        "summarize": "\033[2;33m",
        "grep": "\033[2;35m",
        "grep_file": "\033[2;35m",
        "list_files": "\033[2;37m",
    }
    DEFAULT = "\033[2m"
    MCP = "\033[2;38;5;208m"
    THINKING = "\033[2;3m"
    ERROR = "\033[2;31m"

    @staticmethod
    def tool(name: str, text: str) -> str:
        if Config.color_output_enabled():
            color = Color.TOOL_COLORS.get(name)
            if color is None and "__" in name:
                color = Color.MCP
            return f"{color or Color.DEFAULT}{text}{Color.RESET}"
        return text

    @staticmethod
    def dim(text: str) -> str:
        return Color.tool("default", text)

    @staticmethod
    def thinking(text: str) -> str:
        if Config.color_output_enabled():
            return f"{Color.THINKING}{text}{Color.RESET}"
        return text
