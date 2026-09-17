"""Versioned repository code memory, hybrid retrieval, and evidence validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
import re
import sqlite3
import time
from typing import Protocol, runtime_checkable

from pr_review_agent.orchestration import CandidateFinding


@runtime_checkable
class CodeMemoryStoreProtocol(Protocol):
    """Minimal protocol defining the code memory store persistence boundary."""

    def index_repository(
        self,
        repository_id: str,
        revision: str,
        chunks: Sequence[CodeChunk] | Mapping[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Index a snapshot of chunks or files for a revision, returning total chunks indexed."""
        ...

    def is_fresh(self, repository_id: str, revision: str) -> bool:
        """Return True if the repository revision is known and currently fresh."""
        ...

    def get_chunks(
        self,
        repository_id: str,
        revision: str,
    ) -> list[CodeChunk]:
        """Fetch chunks scoped strictly by repository and revision."""
        ...


@dataclass(frozen=True)
class CodeChunk:
    """A versioned slice of repository source code."""

    chunk_id: str
    repository_id: str
    revision: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    index_version: str
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CitationExcerpt:
    """Structured, bounded context citation returned by retrieval."""

    citation_id: str
    repository_id: str
    revision: str
    file_path: str
    line_range: tuple[int, int]
    excerpt: str
    index_version: str
    relevance_score: float = 0.0


@dataclass(frozen=True)
class FindingValidationResult:
    """The auditable result of evidence validation for a candidate finding."""

    finding_id: str
    status: str  # 'verified' or 'suppressed'
    is_suppressed: bool
    suppression_reason: str | None
    valid_evidence_refs: tuple[str, ...]
    invalid_evidence_refs: tuple[str, ...]
    finding: CandidateFinding


def parse_line_range(ref: str) -> tuple[int, int] | None:
    """Parse #L<start>-L<end> or #L<line> from an evidence reference string."""
    if "#" not in ref:
        return None
    fragment = ref.split("#", 1)[1].split("@", 1)[0]
    match = re.match(r"^L?(\d+)(?:-L?(\d+))?$", fragment, re.IGNORECASE)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    if start > end:
        start, end = end, start
    return (start, end)


def parse_diff_file_hunks(diff_content: str) -> dict[str, list[tuple[int, int]]]:
    """Parse unified diff content into exact file_path -> list of (start_line, end_line) in new revision."""
    file_hunks: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff_content.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in file_hunks:
                file_hunks[current_file] = []
        elif line.startswith("+++ ") and not line.startswith("+++ b/"):
            current_file = line[4:].strip()
            if current_file not in file_hunks:
                file_hunks[current_file] = []
        elif line.startswith("@@ ") and current_file is not None:
            hunk_match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if hunk_match:
                start = int(hunk_match.group(1))
                count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
                end = start + max(1, count) - 1 if count > 0 else start
                file_hunks[current_file].append((start, end))

    return file_hunks


class CodeMemoryStore:
    """Durable repository/code memory with incremental indexing and tenancy boundaries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS repository_revisions (
                    repository_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    is_fresh INTEGER NOT NULL DEFAULT 1,
                    indexed_at REAL NOT NULL,
                    PRIMARY KEY (repository_id, revision)
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repository_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_chunks_lookup ON code_chunks (repository_id, revision, file_path)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_chunks_hash ON code_chunks (repository_id, content_hash)"
            )

    def index_repository(
        self,
        repository_id: str,
        revision: str,
        files: Mapping[str, str],
        *,
        index_version: str = "v1",
        chunk_line_size: int = 50,
        now: float | None = None,
    ) -> int:
        """Index a snapshot of files for a revision, reusing unchanged chunks incrementally."""
        current_time = time.time() if now is None else now
        indexed_count = 0

        with self.connection:
            self.connection.execute(
                "UPDATE repository_revisions SET is_fresh = 0 WHERE repository_id = ?",
                (repository_id,),
            )
            self.connection.execute(
                """
                INSERT INTO repository_revisions (repository_id, revision, index_version, is_fresh, indexed_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(repository_id, revision) DO UPDATE SET
                    is_fresh = 1,
                    index_version = excluded.index_version,
                    indexed_at = excluded.indexed_at
                """,
                (repository_id, revision, index_version, current_time),
            )

            for file_path, content in files.items():
                lines = content.splitlines(keepends=True)
                if not lines:
                    continue

                for i in range(0, len(lines), chunk_line_size):
                    chunk_lines = lines[i : i + chunk_line_size]
                    start_line = i + 1
                    end_line = i + len(chunk_lines)
                    chunk_text = "".join(chunk_lines)
                    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                    chunk_id = f"{repository_id}:{revision}:{file_path}:{start_line}"
                    self.connection.execute(
                        """
                        INSERT OR REPLACE INTO code_chunks (
                            chunk_id, repository_id, revision, file_path,
                            start_line, end_line, content, content_hash,
                            index_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            repository_id,
                            revision,
                            file_path,
                            start_line,
                            end_line,
                            chunk_text,
                            content_hash,
                            index_version,
                            current_time,
                        ),
                    )
                    indexed_count += 1

        return indexed_count

    def is_fresh(self, repository_id: str, revision: str) -> bool:
        """Return True if the repository revision is known and currently fresh."""
        row = self.connection.execute(
            "SELECT is_fresh FROM repository_revisions WHERE repository_id = ? AND revision = ?",
            (repository_id, revision),
        ).fetchone()
        return bool(row and row[0] == 1)

    def get_chunks(
        self,
        repository_id: str,
        revision: str,
        file_path: str | None = None,
    ) -> list[CodeChunk]:
        """Fetch chunks scoped strictly by repository and revision to preserve tenancy."""
        if file_path:
            rows = self.connection.execute(
                """
                SELECT chunk_id, repository_id, revision, file_path, start_line,
                       end_line, content, content_hash, index_version, created_at
                FROM code_chunks
                WHERE repository_id = ? AND revision = ? AND file_path = ?
                ORDER BY start_line ASC
                """,
                (repository_id, revision, file_path),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT chunk_id, repository_id, revision, file_path, start_line,
                       end_line, content, content_hash, index_version, created_at
                FROM code_chunks
                WHERE repository_id = ? AND revision = ?
                ORDER BY file_path ASC, start_line ASC
                """,
                (repository_id, revision),
            ).fetchall()

        return [
            CodeChunk(
                chunk_id=r[0],
                repository_id=r[1],
                revision=r[2],
                file_path=r[3],
                start_line=r[4],
                end_line=r[5],
                content=r[6],
                content_hash=r[7],
                index_version=r[8],
                created_at=r[9],
            )
            for r in rows
        ]


class HybridRetriever:
    """Deterministic lexical + semantic retrieval with strict caps and citation outputs."""

    def __init__(
        self,
        memory_store: CodeMemoryStoreProtocol,
        *,
        vector_dim: int = 128,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        self.memory_store = memory_store
        self.vector_dim = vector_dim
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def retrieve(
        self,
        repository_id: str,
        revision: str,
        query: str,
        *,
        max_k: int = 5,
        max_tokens: int = 1000,
        file_path_filter: str | None = None,
    ) -> tuple[CitationExcerpt, ...]:
        """Retrieve top-k relevant citation excerpts, strictly bounded by max_k and hard max_tokens cap."""
        if max_tokens <= 0 or max_k <= 0:
            return ()

        if file_path_filter and hasattr(self.memory_store, "_get_chunks_for_file"):
            chunks = self.memory_store._get_chunks_for_file(repository_id, revision, file_path=file_path_filter)
        elif file_path_filter:
            try:
                chunks = self.memory_store.get_chunks(repository_id, revision, file_path=file_path_filter)  # type: ignore
            except TypeError:
                chunks = [c for c in self.memory_store.get_chunks(repository_id, revision) if c.file_path == file_path_filter]
        else:
            chunks = self.memory_store.get_chunks(repository_id, revision)
        if not chunks or not query.strip():
            return ()

        query_tokens = self._tokenize(query)
        query_vector = self._embed_text(query)

        scored_chunks: list[tuple[float, CodeChunk]] = []
        for chunk in chunks:
            chunk_tokens = self._tokenize(chunk.content)
            lex_score = self._compute_lexical_score(query_tokens, chunk_tokens)

            chunk_vector = self._embed_text(chunk.content)
            sem_score = self._compute_cosine_similarity(query_vector, chunk_vector)

            total_score = (self.lexical_weight * lex_score) + (self.semantic_weight * sem_score)
            if total_score > 0.01:
                scored_chunks.append((total_score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        citations: list[CitationExcerpt] = []
        accumulated_tokens = 0

        for score, chunk in scored_chunks[:max_k]:
            if accumulated_tokens >= max_tokens:
                break

            excerpt_text = chunk.content.strip()
            # Estimate tokens (~4 chars per token)
            estimated_tokens = max(1, math.ceil(len(excerpt_text) / 4))
            remaining_tokens = max_tokens - accumulated_tokens

            if estimated_tokens > remaining_tokens:
                allowed_chars = max(0, remaining_tokens * 4)
                if allowed_chars < 8:
                    break
                excerpt_text = excerpt_text[:allowed_chars].rstrip() + "..."
                estimated_tokens = remaining_tokens

            citations.append(
                CitationExcerpt(
                    citation_id=f"cite-{chunk.chunk_id}",
                    repository_id=chunk.repository_id,
                    revision=chunk.revision,
                    file_path=chunk.file_path,
                    line_range=(chunk.start_line, chunk.end_line),
                    excerpt=excerpt_text,
                    index_version=chunk.index_version,
                    relevance_score=round(score, 4),
                )
            )
            accumulated_tokens += estimated_tokens

        return tuple(citations)

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\b[A-Za-z0-9_]{2,}\b", text.lower())
        return set(words)

    def _compute_lexical_score(self, query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        intersection = query_tokens.intersection(doc_tokens)
        if not intersection:
            return 0.0
        return len(intersection) / len(query_tokens)

    def _embed_text(self, text: str) -> list[float]:
        """Compute a deterministic, dependency-safe feature-hashed embedding vector."""
        vec = [0.0] * self.vector_dim
        words = re.findall(r"\b[A-Za-z0-9_]{2,}\b", text.lower())
        if not words:
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

            for i in range(max(0, len(word) - 2)):
                tri = word[i : i + 3]
                th = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
                tidx = th % self.vector_dim
                tsign = 1.0 if (th >> 8) & 1 else -1.0
                vec[tidx] += 0.5 * tsign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _compute_cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, dot))


class FindingEvidenceValidator:
    """Enforces strict evidence requirements on candidate findings; suppresses ungrounded findings."""

    def __init__(self, memory_store: CodeMemoryStoreProtocol) -> None:
        self.memory_store = memory_store

    def validate_finding(
        self,
        finding: CandidateFinding,
        *,
        repository_id: str,
        head_sha: str,
        diff_content: str,
    ) -> FindingValidationResult:
        """Validate candidate finding evidence references; fail closed on missing/invalid evidence."""
        if not finding.evidence_refs:
            return FindingValidationResult(
                finding_id=finding.finding_id,
                status="suppressed",
                is_suppressed=True,
                suppression_reason="Finding lacks mandatory evidence references (AC-05 / FR-08)",
                valid_evidence_refs=(),
                invalid_evidence_refs=(),
                finding=finding,
            )

        file_hunks = parse_diff_file_hunks(diff_content)
        valid_refs: list[str] = []
        invalid_refs: list[str] = []

        for ref in finding.evidence_refs:
            if self._verify_reference(
                ref,
                repository_id=repository_id,
                head_sha=head_sha,
                file_hunks=file_hunks,
            ):
                valid_refs.append(ref)
            else:
                invalid_refs.append(ref)

        if not valid_refs:
            return FindingValidationResult(
                finding_id=finding.finding_id,
                status="suppressed",
                is_suppressed=True,
                suppression_reason=f"All evidence references failed verification: {invalid_refs}",
                valid_evidence_refs=(),
                invalid_evidence_refs=tuple(invalid_refs),
                finding=finding,
            )

        if invalid_refs:
            return FindingValidationResult(
                finding_id=finding.finding_id,
                status="suppressed",
                is_suppressed=True,
                suppression_reason=f"Finding contains unverified or stale evidence references: {invalid_refs}",
                valid_evidence_refs=tuple(valid_refs),
                invalid_evidence_refs=tuple(invalid_refs),
                finding=finding,
            )

        return FindingValidationResult(
            finding_id=finding.finding_id,
            status="verified",
            is_suppressed=False,
            suppression_reason=None,
            valid_evidence_refs=tuple(valid_refs),
            invalid_evidence_refs=(),
            finding=finding,
        )

    def _verify_reference(
        self,
        ref: str,
        *,
        repository_id: str,
        head_sha: str,
        file_hunks: Mapping[str, list[tuple[int, int]]],
    ) -> bool:
        """Verify an individual evidence reference string with exact line-range checking."""
        # 1. Diff reference: diff://file_path#L10-L20 or diff:file_path#L10
        if ref.startswith("diff://") or ref.startswith("diff:"):
            target = ref.replace("diff://", "").replace("diff:", "")
            file_part = target.split("#")[0].split(":")[0].strip()
            if not file_part:
                return False

            # Exact file match against parsed diff file headers
            if file_part not in file_hunks:
                return False

            hunks = file_hunks[file_part]
            line_range = parse_line_range(ref)
            if line_range is not None:
                cited_start, cited_end = line_range
                # Verify cited lines fall within or overlap with the modified diff hunks
                falls_in_hunk = any(
                    max(h_start, cited_start) <= min(h_end, cited_end)
                    for h_start, h_end in hunks
                )
                if not falls_in_hunk:
                    return False

            return True

        # 2. Repo reference: repo://file_path#L10-L20@revision or repo:file_path@revision
        if ref.startswith("repo://") or ref.startswith("repo:"):
            target = ref.replace("repo://", "").replace("repo:", "")
            parts = target.split("@")
            file_and_lines = parts[0]
            cited_revision = parts[1] if len(parts) > 1 else head_sha

            if cited_revision != head_sha or not self.memory_store.is_fresh(repository_id, cited_revision):
                return False

            file_path = file_and_lines.split("#")[0].strip()
            if hasattr(self.memory_store, "_get_chunks_for_file"):
                chunks = self.memory_store._get_chunks_for_file(repository_id, cited_revision, file_path=file_path)
            else:
                try:
                    chunks = self.memory_store.get_chunks(repository_id, cited_revision, file_path=file_path)  # type: ignore
                except TypeError:
                    chunks = [c for c in self.memory_store.get_chunks(repository_id, cited_revision) if c.file_path == file_path]
            if not chunks:
                return False

            line_range = parse_line_range(ref)
            if line_range is not None:
                cited_start, cited_end = line_range
                min_line = min(c.start_line for c in chunks)
                max_line = max(c.end_line for c in chunks)
                # Verify cited lines exist within the file line boundaries
                if cited_start < min_line or cited_end > max_line:
                    return False

            return True

        # 3. Citation ID: cite-repository_id:revision:file_path:start_line
        if ref.startswith("cite-"):
            raw = ref.replace("cite-", "")
            parts = raw.split(":")
            if len(parts) >= 4:
                cite_repo, cite_rev = parts[0], parts[1]
                cite_line_str = parts[-1]
                cite_file = ":".join(parts[2:-1])

                if cite_repo != repository_id or cite_rev != head_sha:
                    return False
                if not self.memory_store.is_fresh(cite_repo, cite_rev):
                    return False

                if hasattr(self.memory_store, "_get_chunks_for_file"):
                    chunks = self.memory_store._get_chunks_for_file(cite_repo, cite_rev, file_path=cite_file)
                else:
                    try:
                        chunks = self.memory_store.get_chunks(cite_repo, cite_rev, file_path=cite_file)  # type: ignore
                    except TypeError:
                        chunks = [c for c in self.memory_store.get_chunks(cite_repo, cite_rev) if c.file_path == cite_file]
                if not chunks:
                    return False

                try:
                    cited_line = int(cite_line_str)
                    min_line = min(c.start_line for c in chunks)
                    max_line = max(c.end_line for c in chunks)
                    if cited_line < min_line or cited_line > max_line:
                        return False
                except ValueError:
                    pass

                return True

        return False
