import os
import patch
from threading import Lock

# ---------------------------------------------------------------------------
# 常量：有效代码后缀白名单 + 忽略目录/文件黑名单
# 想扩展支持的语言，改这里即可
# ---------------------------------------------------------------------------
VALID_CODE_EXTS = {
    # 主流后端/系统语言
    ".go", ".py", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh",
    ".rs", ".java", ".kt", ".scala", ".swift", ".m", ".mm",
    # 前端
    ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
    # 脚本
    ".sh", ".bash", ".zsh", ".rb", ".php", ".lua", ".pl",
    # 数据/协议/配置（跟代码强相关的）
    ".proto", ".thrift", ".sql", ".graphql",
}

IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".DS_Store",
    "node_modules", "vendor", "dist", "build", "target", "out",
    "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", ".turbo", ".cache", "coverage", ".nyc_output",
    "bin", "obj",
}

# 生成物 / 特别大的锁文件 / 二进制，直接屏蔽
IGNORE_FILE_NAMES = {
    "go.sum", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "composer.lock", "Pipfile.lock", "poetry.lock",
}


def get_valid_exts() -> set:
    """暴露白名单集合，方便外部临时扩展。"""
    return VALID_CODE_EXTS


# ---------------------------------------------------------------------------
# 目录图谱：进程级单例，按项目根路径分桶缓存
# ---------------------------------------------------------------------------
class _ProjectGraph:
    """
    项目目录图谱。首次进入某个项目根路径时，全量扫描一次并构建索引；
    之后 list_project_directory 都直接从索引里取"某一层"的结果，
    不再触碰磁盘，也绝不会一次性把全部文件抛给模型。

    数据结构：
        self._trees[root] = {
            "root": root,
            "dirs": {abs_dir_path: {
                "path": abs_dir_path,
                "child_dirs": [abs_subdir_path, ...],   # 直接子目录
                "child_files": [abs_file_path, ...],    # 直接子文件（已过滤）
                "descendant_file_count": int,            # 递归后代文件总数
                "ext_stats": {".go": n, ".py": m, ...},  # 递归后代按扩展名统计
            }}
        }
    """

    def __init__(self):
        self._trees = {}
        self._lock = Lock()

    def _scan(self, root: str) -> dict:
        root = os.path.abspath(root)
        dirs_map = {}

        # 用 topdown=True 才能就地裁剪 dirs
        for cur_root, subdirs, files in os.walk(root, topdown=True):
            # 就地裁剪：屏蔽黑名单目录 + 隐藏目录，阻止 os.walk 下钻
            subdirs[:] = sorted([
                d for d in subdirs
                if d not in IGNORE_DIRS and not d.startswith(".")
            ])

            valid_files = []
            for f in sorted(files):
                if f in IGNORE_FILE_NAMES:
                    continue
                if f.startswith("."):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext not in VALID_CODE_EXTS:
                    continue
                valid_files.append(os.path.join(cur_root, f))

            dirs_map[cur_root] = {
                "path": cur_root,
                "child_dirs": [os.path.join(cur_root, d) for d in subdirs],
                "child_files": valid_files,
                "descendant_file_count": 0,
                "ext_stats": {},
            }

        # 自底向上聚合 descendant_file_count / ext_stats
        # 用路径长度倒序确保先算叶子
        for d in sorted(dirs_map.keys(), key=lambda p: -len(p)):
            node = dirs_map[d]
            cnt = len(node["child_files"])
            stats = {}
            for f in node["child_files"]:
                ext = os.path.splitext(f)[1].lower()
                stats[ext] = stats.get(ext, 0) + 1
            for child_dir in node["child_dirs"]:
                child_node = dirs_map.get(child_dir)
                if not child_node:
                    continue
                cnt += child_node["descendant_file_count"]
                for k, v in child_node["ext_stats"].items():
                    stats[k] = stats.get(k, 0) + v
            node["descendant_file_count"] = cnt
            node["ext_stats"] = stats

        return {"root": root, "dirs": dirs_map}

    def get_or_build(self, root: str) -> dict:
        root = os.path.abspath(root)
        with self._lock:
            if root not in self._trees:
                self._trees[root] = self._scan(root)
            return self._trees[root]

    def find_owning_tree(self, path: str) -> dict | None:
        """给定任意路径，找到已缓存的、包含它的项目图谱。"""
        path = os.path.abspath(path)
        with self._lock:
            for root, tree in self._trees.items():
                if path == root or path.startswith(root.rstrip(os.sep) + os.sep):
                    return tree
        return None


_GRAPH = _ProjectGraph()


def build_project_graph(root: str) -> dict:
    """
    显式为项目根目录构建（或复用）图谱。
    可在任务开始时主动调用一次；也可以不调用，list_project_directory 首次访问时会自动构建。
    返回值包含 root 和汇总统计，供上层日志展示。
    """
    tree = _GRAPH.get_or_build(root)
    root_node = tree["dirs"].get(tree["root"], {})
    return {
        "root": tree["root"],
        "total_files": root_node.get("descendant_file_count", 0),
        "ext_stats": root_node.get("ext_stats", {}),
        "top_level_dirs": [os.path.basename(d) for d in root_node.get("child_dirs", [])],
    }


# ---------------------------------------------------------------------------
# 对外工具
# ---------------------------------------------------------------------------
def list_project_directory(path: str, max_entries: int = 200) -> dict:
    """
    List ONE level of a project subdirectory (never recurses).

    IMPORTANT for the calling model:
      - Use this to explore the project structure step by step.
      - The returned `dirs[*].path` are DIRECTORIES: to look inside one of them,
        call `list_project_directory` again with that path.
      - The returned `files[*].path` are FILES: to read a file's content,
        call `read_file` with that path.
      - NEVER pass a `dirs[*].path` to `read_file`.

    Sub-directories carry `file_count` and `ext_stats` so you can decide whether
    to drill down without over-fetching.

    :param path: absolute path of a directory inside the project
    :param max_entries: max entries for dirs / files lists respectively
    """
    path = os.path.abspath(path)

    # 先在已有图谱里找归属；找不到就把 path 当项目根重新扫描
    tree = _GRAPH.find_owning_tree(path)
    if tree is None:
        tree = _GRAPH.get_or_build(path)

    node = tree["dirs"].get(path)
    if node is None:
        # path 是文件、或不在图谱内（可能被黑名单屏蔽了）
        return {
            "path": path,
            "dirs": [],
            "files": [],
            "note": "path not indexed (maybe a file, or filtered by ignore list, or not a directory in project)",
            "usage": "If you intended to read a file, call read_file(path) instead.",
        }

    dirs_info = []
    for d in node["child_dirs"][:max_entries]:
        sub = tree["dirs"].get(d, {})
        dirs_info.append({
            "path": d,
            "name": os.path.basename(d),
            "type": "dir",
            "file_count": sub.get("descendant_file_count", 0),
            "ext_stats": sub.get("ext_stats", {}),
        })

    files_info = [
        {"path": f, "name": os.path.basename(f), "type": "file"}
        for f in node["child_files"][:max_entries]
    ]

    return {
        "path": path,
        "dirs": dirs_info,
        "files": files_info,
        "truncated_dirs": len(node["child_dirs"]) > max_entries,
        "truncated_files": len(node["child_files"]) > max_entries,
        "usage": (
            "Call list_project_directory on any dirs[*].path to go deeper. "
            "Call read_file on any files[*].path to read its content. "
            "Do NOT pass a dirs[*].path to read_file."
        ),
    }


def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
    max_bytes: int = 200_000,
    with_line_numbers: bool = True,
) -> str:
    """
    Read a FILE, optionally limited to a specific line range.

    STRONG RECOMMENDATION: to save tokens, do NOT read the whole file unless
    you really need it. Prefer one of the following patterns:
      * Call `grep_file(pattern, path)` first to locate the interesting lines,
        then call `read_file(path, start_line=..., end_line=...)` on that range.
      * If you already know the target line (e.g. from an error stack, or from
        SQL parse result's `position`), pass `start_line` and `end_line` so we
        only load a small window (e.g. 50 lines around it).

    Arguments:
      path            : absolute path to a regular FILE (not a directory).
      start_line      : 1-based inclusive start line. Default 1.
      end_line        : 1-based inclusive end line. Default None = until EOF
                        or until max_bytes is hit.
      max_bytes       : hard cap on returned bytes (safety net).
      with_line_numbers: if True (default), each returned line is prefixed with
                        its absolute 1-based line number and a colon, e.g.
                            "  123: some code here"
                        so that any subsequent `write_file` diff can use the
                        correct hunk line numbers. Set to False only if you
                        need the raw content (rare).

    Guardrails:
      - `path` MUST be a file. Directories → call `list_project_directory`.
      - Non-existent path → returns an error string with guidance.
    """
    if not os.path.exists(path):
        return (
            f"[read_file error] path does not exist: {path!r}. "
            f"If you want to explore a directory, call list_project_directory instead."
        )
    if os.path.isdir(path):
        return (
            f"[read_file error] path is a DIRECTORY, not a file: {path!r}. "
            f"Please call list_project_directory({path!r}) to see its contents, "
            f"then pick a file path from the 'files' field and call read_file again."
        )
    if not os.path.isfile(path):
        return f"[read_file error] path is not a regular file: {path!r}"

    # 归一化行范围参数
    try:
        start_line = int(start_line) if start_line is not None else 1
    except (TypeError, ValueError):
        start_line = 1
    if start_line < 1:
        start_line = 1
    if end_line is not None:
        try:
            end_line = int(end_line)
        except (TypeError, ValueError):
            end_line = None
        if end_line is not None and end_line < start_line:
            return (
                f"[read_file error] end_line ({end_line}) < start_line ({start_line}). "
                f"Please pass a valid inclusive range."
            )

    try:
        total_size = os.path.getsize(path)
    except OSError as e:
        return f"[read_file error] {e}"

    # 按行流式读取，避免一次性把大文件全塞进内存
    picked_lines: list[str] = []
    total_bytes = 0
    total_lines = 0
    truncated_by_bytes = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            for idx, raw in enumerate(file, start=1):
                total_lines = idx
                if idx < start_line:
                    continue
                if end_line is not None and idx > end_line:
                    # 记录总行数需要继续扫，但避免把剩余行也塞进 picked
                    # 为节省时间直接 break——只是总行数会显示为"至少 idx-1"
                    total_lines = idx - 1
                    break
                # 去掉行尾换行以便统一加编号
                line_body = raw.rstrip("\n")
                if with_line_numbers:
                    piece = f"{idx:>6}: {line_body}\n"
                else:
                    piece = line_body + "\n"
                b = len(piece.encode("utf-8", errors="ignore"))
                if total_bytes + b > max_bytes:
                    truncated_by_bytes = True
                    break
                picked_lines.append(piece)
                total_bytes += b
    except OSError as e:
        return f"[read_file error] {e}"

    if not picked_lines:
        return (
            f"[read_file] no lines returned for {path} "
            f"with range start_line={start_line}, end_line={end_line} "
            f"(file has ~{total_lines} lines, {total_size} bytes)."
        )

    header = (
        f"# file: {path}\n"
        f"# range: lines {start_line}..{end_line if end_line is not None else 'EOF'}\n"
        f"# file_stats: total_bytes={total_size}"
    )
    if truncated_by_bytes:
        header += f"; truncated_by_max_bytes={max_bytes}"
    header += "\n"
    return header + "".join(picked_lines)


def grep_file(
    pattern: str,
    path: str,
    context: int = 3,
    max_matches: int = 20,
    max_bytes: int = 60_000,
    ignore_case: bool = False,
    include_exts: list | None = None,
) -> str:
    """
    Search for a regex `pattern` inside a FILE or a DIRECTORY, and return each
    match together with a few context lines around it.

    USE THIS INSTEAD OF READING WHOLE FILES when you only need to locate a
    symbol / string / SQL / config key. It is dramatically cheaper in tokens.

    Arguments:
      pattern     : Python regular expression string (case-sensitive by default).
                    Escape regex metacharacters if you want a literal match.
      path        : absolute path. Either:
                      * a FILE  → grep only this file, or
                      * a DIR   → grep every code file under it (whitelisted
                                  by VALID_CODE_EXTS, ignoring vendor/.git/...).
      context     : number of lines of context to include BEFORE and AFTER
                    each matching line. Default 3.
      max_matches : hard cap on how many matches are returned in total across
                    all files. Default 20. If exceeded, output is truncated and
                    a "truncated: true" hint is added.
      max_bytes   : hard cap on returned bytes. Safety net for very wide
                    patterns. Default 60_000.
      ignore_case : case-insensitive matching. Default False.
      include_exts: optional list like [".go", ".py"] to further restrict which
                    extensions to search in DIRECTORY mode. None = use the
                    project's default code-extension whitelist.

    Output format (a plain string):
        # grep pattern=... path=... ignore_case=... context=...
        --- /abs/path/to/file.go ---
           118: some code
           119: some code
        >> 120: THE MATCHING LINE
           121: some code
           122: some code
        --- /abs/path/to/another.go ---
           ...
        # matches_found: 5, truncated: false

    Follow-up pattern for the model:
      1. Call grep_file to locate line numbers.
      2. Call read_file(path, start_line=match-20, end_line=match+40) for a
         precise window if you need more surrounding context.
      3. Only then call write_file / create_file.
    """
    import re

    if not pattern:
        return "[grep_file error] pattern must not be empty."
    if not os.path.exists(path):
        return (
            f"[grep_file error] path does not exist: {path!r}. "
            f"Call list_project_directory first to discover valid paths."
        )
    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"[grep_file error] invalid regex {pattern!r}: {e}"

    # 决定要扫描的文件集合
    files_to_scan: list[str] = []
    if os.path.isfile(path):
        files_to_scan = [path]
    else:
        # 目录：走项目图谱，只扫代码白名单
        tree = _GRAPH.find_owning_tree(path)
        if tree is None:
            tree = _GRAPH.get_or_build(path)
        exts = set(include_exts) if include_exts else set(VALID_CODE_EXTS)
        for d, node in tree["dirs"].items():
            # d 必须在 path 之下（或等于 path）
            if not (d == path or d.startswith(os.path.abspath(path).rstrip(os.sep) + os.sep)):
                continue
            for f in node["child_files"]:
                ext = os.path.splitext(f)[1].lower()
                if ext in exts:
                    files_to_scan.append(f)

    if not files_to_scan:
        return (
            f"[grep_file] no code files to scan under {path!r} "
            f"(check include_exts / whitelist)."
        )

    total_matches = 0
    total_bytes = 0
    truncated = False
    out_parts: list[str] = []
    out_parts.append(
        f"# grep pattern={pattern!r} path={path!r} "
        f"ignore_case={ignore_case} context={context}\n"
    )

    for f in files_to_scan:
        if total_matches >= max_matches or total_bytes >= max_bytes:
            truncated = True
            break
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                lines = fp.readlines()
        except OSError:
            continue

        file_hits: list[int] = [i for i, line in enumerate(lines) if regex.search(line)]
        if not file_hits:
            continue

        section: list[str] = [f"--- {f} ---\n"]
        # 合并相邻/重叠的上下文窗口，避免同一片区域被重复渲染
        windows: list[tuple[int, int, set]] = []  # (start_idx, end_idx, match_idx_set)
        for hit in file_hits:
            s = max(0, hit - context)
            e = min(len(lines) - 1, hit + context)
            if windows and s <= windows[-1][1] + 1:
                ps, pe, pm = windows[-1]
                pm.add(hit)
                windows[-1] = (ps, max(pe, e), pm)
            else:
                windows.append((s, e, {hit}))

        for (s, e, ms) in windows:
            if total_matches >= max_matches or total_bytes >= max_bytes:
                truncated = True
                break
            for i in range(s, e + 1):
                marker = ">>" if i in ms else "  "
                # 1-based 行号
                piece = f"{marker} {i + 1:>6}: {lines[i].rstrip(chr(10))}\n"
                if total_bytes + len(piece.encode("utf-8", errors="ignore")) > max_bytes:
                    truncated = True
                    break
                section.append(piece)
                total_bytes += len(piece.encode("utf-8", errors="ignore"))
            # 每个窗口后加空行分隔
            section.append("\n")
            total_bytes += 1
            total_matches += len(ms)
            if total_matches >= max_matches:
                truncated = True
                break

        out_parts.extend(section)

    out_parts.append(
        f"# matches_found: {total_matches}, truncated: {'true' if truncated else 'false'}\n"
    )
    if truncated:
        out_parts.append(
            "# hint: narrow your pattern, or increase max_matches / max_bytes, "
            "or scope path to a smaller subdirectory.\n"
        )
    return "".join(out_parts)


def write_file(path: str, diff: str):
    """
    Apply a unified-diff patch to an existing FILE and write the result back.

    IMPORTANT for the calling model:
      - `path` MUST be an absolute path pointing to an existing regular FILE
        (the file you want to modify). DO NOT pass a directory path here.
        To explore a directory, call `list_project_directory` first, then pick
        the target file from its `files[*].path`.
      - If the target file does NOT exist yet, DO NOT use write_file — call
        `create_file(path, content)` instead.
      - `diff` MUST be a unified-diff string, i.e. the same textual format
        produced by `git diff` / `diff -u`. It should contain the standard
        headers and hunks, for example:

            --- a/path/to/file.go
            +++ b/path/to/file.go
            @@ -10,7 +10,7 @@
             unchanged line
            -removed line
            +added line
             unchanged line

        The diff is applied to the current content of `path`; line numbers in
        the hunk headers must match the current file. Do NOT pass the full new
        file content here — only the diff.

    :param path: absolute path of the file to modify (must be an existing file)
    :param diff: unified-diff string (git-diff style) describing the change
    """
    if os.path.isdir(path):
        return (
            f"[write_file error] path is a DIRECTORY, not a file: {path!r}. "
            f"Pass the absolute path of the specific file you want to modify."
        )
    if not os.path.isfile(path):
        return (
            f"[write_file error] file does not exist: {path!r}. "
            f"write_file can only patch an existing file. "
            f"If you want to CREATE a new file, call create_file(path, content) instead."
        )

    # 注意：第三方 patch 库（PyPI 的 patch / patch-ng）内部按 bytes 处理，
    # 名字虽然叫 fromstring，实际要求 bytes 参数，否则会抛：
    #   TypeError: a bytes-like object is required, not 'str'
    if isinstance(diff, str):
        diff_bytes = diff.encode("utf-8", errors="replace")
    elif isinstance(diff, (bytes, bytearray)):
        diff_bytes = bytes(diff)
    else:
        return (
            f"[write_file error] diff must be a str or bytes containing a unified-diff, "
            f"got {type(diff).__name__}."
        )

    # 规范化换行，避免混合 \r\n / \r 让 patch 库解析失败
    diff_bytes = diff_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # 一些模型会漏掉结尾换行，patch 库对此比较敏感
    if not diff_bytes.endswith(b"\n"):
        diff_bytes += b"\n"

    try:
        p = patch.fromstring(diff_bytes)
    except Exception as e:
        return (
            f"[write_file error] failed to parse diff: {e}. "
            f"The `diff` argument must be a valid unified-diff string, e.g.\n"
            f"    --- a/relative/path\n"
            f"    +++ b/relative/path\n"
            f"    @@ -L,N +L,N @@\n"
            f"     unchanged\n"
            f"    -removed\n"
            f"    +added\n"
        )
    if not p:
        # patch 库解析失败会返回 False，而不是抛异常
        return (
            f"[write_file error] diff is not a valid unified-diff. "
            f"Make sure it starts with '--- a/...' and '+++ b/...' headers, "
            f"contains '@@ -L,N +L,N @@' hunk headers, and that line numbers "
            f"match the CURRENT content of {path!r} (call read_file first to "
            f"get correct line numbers)."
        )

    # patch 库期望 bytes 原文
    try:
        with open(path, "rb") as _f:
            original_bytes = _f.read()
    except OSError as e:
        return f"[write_file error] failed to read original file {path!r}: {e}"

    result = p.apply(original_bytes)
    if result is False or result is None:
        return (
            f"[write_file error] failed to apply diff to {path!r}. "
            f"This usually means the hunk context/line numbers do not match "
            f"the current file. Re-read the target region with read_file to "
            f"get up-to-date line numbers, then regenerate the diff."
        )

    try:
        with open(path, "wb") as f:
            f.write(result)
    except OSError as e:
        return f"[write_file error] failed to write file {path!r}: {e}"

    # 落盘证据
    import sys
    print(
        f"[write_file DISK] patched {path} "
        f"({len(original_bytes)} -> {len(result)} bytes)",
        file=sys.stderr,
        flush=True,
    )
    return f"[write_file ok] patched {path} ({len(original_bytes)} -> {len(result)} bytes)"


def create_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Create a NEW file with the given full content.

    IMPORTANT for the calling model:
      - Use this tool when you need to CREATE a brand-new file that does not
        exist yet. For modifying an EXISTING file, use `write_file` (which
        applies a unified diff) instead.
      - `path` MUST be an absolute path pointing to the file to create (not a
        directory). Missing parent directories will be created automatically.
      - `content` is the FULL text content of the new file (no diff format).
      - By default this tool refuses to overwrite an existing file. If you
        really intend to replace an existing file's content wholesale, pass
        `overwrite=True` — but prefer `write_file` for edits so the change is
        expressed as a diff.

    :param path: absolute path of the file to create
    :param content: full text content to write into the file
    :param overwrite: if True, allow overwriting an existing file (default False)
    :return: a short human-readable status string
    """
    if os.path.isdir(path):
        return (
            f"[create_file error] path is a DIRECTORY, not a file: {path!r}. "
            f"Pass the absolute path of the file you want to create."
        )
    if os.path.exists(path) and not overwrite:
        return (
            f"[create_file error] file already exists: {path!r}. "
            f"To modify it, call write_file(path, diff). "
            f"To replace it wholesale, call create_file(path, content, overwrite=True)."
        )

    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"[create_file error] failed to create parent directory {parent!r}: {e}"

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"[create_file error] failed to write file {path!r}: {e}"

    # 直接 stderr 打印一次，作为"真的写盘了"的强证据，方便与模型口嗨对齐
    import sys
    print(f"[create_file DISK] wrote {path} ({len(content)} bytes)", file=sys.stderr, flush=True)

    # 新增/覆盖文件后使已缓存的项目图谱失效，避免后续 list_project_directory 看不到新文件
    tree = _GRAPH.find_owning_tree(path)
    if tree is not None:
        with _GRAPH._lock:
            _GRAPH._trees.pop(tree["root"], None)

    action = "overwritten" if overwrite else "created"
    return f"[create_file ok] {action} {path} ({len(content)} bytes)"


if __name__ == "__main__":
    import json
    summary = build_project_graph("/Users/src")
    print("== project graph summary ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("== list root ==")
    print(json.dumps(list_project_directory(summary["root"]), ensure_ascii=False, indent=2))
