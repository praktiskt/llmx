import re
from collections import OrderedDict

from .config import Config


class Cache:
    MAX_ENTRIES = 50
    _storage: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def new_id() -> str:
        while True:
            file_id = Config.generate_file_id()
            if file_id not in Cache._storage:
                return file_id

    @staticmethod
    def store(file_id: str, content: str) -> None:
        wrapped_lines = []
        for line in content.splitlines():
            line = re.sub(r"(data:[^,]+,)[^)\s]+", r"\1[TRUNCATED]", line)

            if len(line) <= 200:
                wrapped_lines.append(line)
            else:
                start = 0
                while start < len(line):
                    chunk = line[start : start + 200]
                    last_space = chunk.rfind(" ")
                    if last_space > 0:
                        wrapped_lines.append(chunk[:last_space])
                        start += last_space + 1
                    else:
                        wrapped_lines.append(chunk)
                        start += 200
        Cache._storage[file_id] = "\n".join(wrapped_lines)
        Cache._storage.move_to_end(file_id)
        while len(Cache._storage) > Cache.MAX_ENTRIES:
            Cache._storage.popitem(last=False)

    @staticmethod
    def get(file_id: str) -> str | None:
        content = Cache._storage.get(file_id)
        if content is not None:
            Cache._storage.move_to_end(file_id)
        return content

    @staticmethod
    def maybe_store_large(content: str) -> str:
        """If content exceeds MAX_TOOL_RESULT_CHARS, store as memory:// and return reference."""
        if len(content) > Config.MAX_TOOL_RESULT_CHARS:
            file_id = Cache.new_id()
            Cache.store(file_id, content)
            return f"Stored as memory://{file_id} ({len(content)} chars). Use read_file with sources=['memory://{file_id}'] to read or summarize."
        return content

    @staticmethod
    def _format_read(
        label: str,
        content: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        lines = content.splitlines()
        total_lines = len(lines)

        if offset is None:
            offset = 1
        if offset < 1:
            offset = 1

        if limit is None:
            limit = 50
        if limit < 1:
            limit = 1

        start = offset - 1
        end = start + limit

        selected = lines[start:end]
        result = "\n".join(f"{i + offset}: {line}" for i, line in enumerate(selected))

        header = (
            f"File {label} (lines {offset}-{min(end, total_lines)} of {total_lines})\n"
        )
        return header + result

    @staticmethod
    def _grep_content(
        label: str,
        content: str,
        pattern: str,
        is_regex: bool = False,
        ignore_case: bool = False,
        context: int = 0,
    ) -> str:
        lines = content.splitlines()
        total_lines = len(lines)

        flags = re.IGNORECASE if ignore_case else 0
        if is_regex:
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: invalid regex: {e}"

            def matcher(line: str, regex=regex) -> bool:
                return regex.search(line) is not None

        elif ignore_case:
            pattern_lower = pattern.lower()

            def matcher(line: str, pattern_lower=pattern_lower) -> bool:
                return pattern_lower in line.lower()

        else:

            def matcher(line: str, pattern=pattern) -> bool:
                return pattern in line

        matched_indices = set()
        for i, line in enumerate(lines):
            if matcher(line):
                matched_indices.add(i)

        if not matched_indices:
            return f'File {label}: no matches for "{pattern}"'

        all_matched_indices = matched_indices.copy()

        if context > 0:
            context_indices = set()
            for idx in matched_indices:
                for j in range(
                    max(0, idx - context), min(total_lines, idx + context + 1)
                ):
                    context_indices.add(j)
            matched_indices = context_indices

        sorted_indices = sorted(matched_indices)

        groups = []
        current_group = []
        for i, idx in enumerate(sorted_indices):
            if not current_group or idx == sorted_indices[i - 1] + 1:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        if current_group:
            groups.append(current_group)

        output_lines = [
            f'File {label}: {len(all_matched_indices)} matches for "{pattern}"'
        ]
        match_count = 0
        truncated = False

        for group in groups:
            if truncated:
                break
            output_lines.append("--")
            for idx in group:
                if match_count >= Config.GREP_MAX_MATCHES:
                    truncated = True
                    break
                line_num = idx + 1
                prefix = ">" if idx in all_matched_indices else " "
                output_lines.append(f"{prefix}{line_num}: {lines[idx]}")
                match_count += 1

        if len(all_matched_indices) > Config.GREP_MAX_MATCHES:
            output_lines.append("--")
            output_lines.append(
                f"... {len(all_matched_indices) - Config.GREP_MAX_MATCHES} more matches not shown"
            )

        return "\n".join(output_lines)

    @staticmethod
    def read(
        file_ids: list[str], offset: int | None = None, limit: int | None = None
    ) -> str:
        results = []
        for file_id in file_ids:
            validation_error = Config.validate_file_id(file_id)
            if validation_error:
                results.append(f"Error: {validation_error}")
                continue

            content = Cache.get(file_id)
            if content is None:
                results.append(f"Error: file {file_id} not found")
                continue

            results.append(Cache._format_read(file_id, content, offset, limit))

        return "\n\n---\n\n".join(results)

    @staticmethod
    def grep(
        file_ids: list[str],
        pattern: str,
        is_regex: bool = False,
        ignore_case: bool = False,
        context: int = 0,
    ) -> str:
        results = []
        for file_id in file_ids:
            validation_error = Config.validate_file_id(file_id)
            if validation_error:
                results.append(f"Error: {validation_error}")
                continue

            content = Cache.get(file_id)
            if content is None:
                results.append(f"Error: file {file_id} not found")
                continue

            results.append(
                Cache._grep_content(
                    file_id, content, pattern, is_regex, ignore_case, context
                )
            )

        return "\n\n---\n\n".join(results)
