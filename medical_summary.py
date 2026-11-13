"""Opt-in local-only medical document summarizer. CLI: python medical_summary.py < input.json.

This is a draft generator, not a medical decision system. No user content is logged.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

MAX_INPUT_CHARS = 65536
CHUNK_SIZE = 750
MIN_CHUNK_SIZE = 280


def _use_32k_context() -> bool:
    # Opt in only after the shared llama.cpp server is restarted at -c 32768.
    return os.environ.get("FINDINGS_CONTEXT_PROFILE", "legacy") == "32k"


def _source_chunk_size() -> int:
    return 1500 if _use_32k_context() else CHUNK_SIZE


def _direct_input_limit() -> int:
    return 16000 if _use_32k_context() else DIRECT_MAX_CHARS


def _comparison_batch_limit() -> int:
    return 8000 if _use_32k_context() else MAX_COMPARISON_BATCH_CHARS

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {"type": "array", "items": {"type": "string"}},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "site": {"type": "string"},
                    "status": {"type": "string", "enum": ["normal", "abnormal", "uncertain"]},
                    "description": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["site", "status", "description", "evidence"],
                "additionalProperties": False,
            },
        },
        "measurements": {"type": "array", "items": {"type": "string"}},
        "treatments_and_followup": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnoses", "findings", "measurements", "treatments_and_followup", "uncertainties"],
    "additionalProperties": False,
}

RESTORE_SYSTEM = """Ты — система интеллектуального восстановления текста медицинских документов после OCR-распознавания.

Твоя задача — восстановить максимально точный, связный и читаемый текст оригинального медицинского документа.

Исправляй ошибки распознавания символов, искажённые медицинские термины, очевидные опечатки, некорректные переносы, пробелы и пунктуацию. Используй медицинский и языковой контекст для восстановления очевидно повреждённых слов.

Сохраняй исходную структуру документа, диагнозы, результаты исследований, количественные показатели, даты, единицы измерения, названия препаратов и дозировки.

Не сокращай текст, не добавляй медицинских интерпретаций и не меняй смысл исходных утверждений.

Если числовое значение, дозировка или термин не могут быть восстановлены однозначно, сохраняй исходную запись. Не подменяй неизвестные значения предполагаемыми.

Верни исключительно восстановленный текст документа. Не включай служебные инструкции, заголовки запроса, комментарии или пояснения."""

EXTRACT_SYSTEM = """Ты — система медицинского семантического анализа документов.

Проанализируй предоставленный фрагмент и выдели клинически значимую информацию, сохраняя её исходный медицинский смысл.

Диагнозы. Выделяй только заболевания и диагностические состояния, прямо указанные в документе. Различай диагноз, с которым пациент направлен на исследование, и диагноз или заключение, установленное по результатам исследования. Не считай отдельные измерения, симптомы или описания нормальных показателей диагнозами.

Находки. Выделяй все клинически значимые результаты исследования. Отдельно учитывай обнаруженные отклонения и явно зафиксированные нормальные результаты. Если в одном органе присутствует патологическая находка, а в другом отклонений нет, сохрани оба факта, не смешивая их.

Измерения. Сохраняй значимые количественные показатели с единицами измерения. Учитывай анатомическую локализацию и сторону органа. Не приписывай числовым показателям клиническую интерпретацию, которая отсутствует в документе.

Лечение и рекомендации. Если документ содержит назначения или рекомендации, извлекай их как исторические сведения данного документа, не превращая в новые медицинские назначения.

Неопределённости. Указывай только те неоднозначности исходного текста, которые способны изменить клинический смысл. Отсутствие возраста, анамнеза или иных необязательных сведений не является ошибкой документа.

Не добавляй информацию из общих медицинских знаний. Не придумывай диагнозов, патологических изменений или нормальных результатов, которых нет в источнике.

Для каждой находки укажи подтверждающий фрагмент исходного текста.

Верни данные в соответствии с предоставленной JSON-схемой."""

COMMENT_SYSTEM = """Ты — система подготовки кратких медицинских комментариев к результатам обследований для электронной медицинской истории пациента.

На основании извлечённых фактов сформируй связное, профессионально написанное краткое резюме документа.

Отрази основное содержание исследования и все клинически значимые результаты.

Если в документе указан диагноз, с которым пациент направлен на обследование, отделяй его от находок, установленных при исследовании.

В первую очередь описывай выявленные патологические изменения, указывая их локализацию, характеристики и существенные измерения.

Обязательно сохраняй клинически значимые нормальные результаты, особенно если они относятся к другим органам, областям или позволяют понять общий результат исследования.

Если исследование не выявило отклонений, прямо отрази этот результат.

Не повторяй одни и те же находки в разных формулировках. Не превращай измерения в диагнозы. Не дополняй документ предположениями, новыми назначениями или рекомендациями.

Пиши на грамотном русском медицинском языке. Сохраняй точность терминов и клинического смысла.

Комментарий должен быть настолько кратким, насколько возможно без потери клинически значимой информации.

Верни только готовый комментарий без заголовков, пояснений и повторения исходных данных в виде списка."""


class SummaryError(Exception):
    """Safe-to-print operational error; never contains clinical source text."""


def clean_source(raw: str, kind: str) -> str:
    if kind not in ("html", "text", "auto"):
        raise SummaryError("INVALID_FORMAT")
    if kind == "html" or (kind == "auto" and re.search(
        r"<(?:html|body|div|p|table|tr|td|br|section)\b", raw, re.I
    )):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head", "noscript"]):
            tag.decompose()
        for tag in soup.find_all("br"):
            tag.replace_with("\n")
        for tag in soup.find_all(["tr", "p", "div"]):
            tag.append("\n")
        for tag in soup.find_all(["td", "th"]):
            tag.append(" | ")
        raw = soup.get_text(separator=" ", strip=False)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def split_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Prefer line/word boundaries, preserve content other than outer whitespace."""
    if not text.strip():
        return []
    result = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            candidate = max(text.rfind("\n", pos + size // 3, end),
                            text.rfind(" ", pos + size // 3, end))
            if candidate > pos:
                end = candidate + 1
        part = text[pos:end].strip()
        if part:
            result.append(part)
        pos = end
    return result


def _request(messages: list[dict], *, schema: dict | None = None) -> tuple[str, str]:
    url = os.environ.get("LLM_URL", "http://llm:8080/v1/chat/completions")
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("llm", "localhost", "127.0.0.1"):
        raise SummaryError("LOCAL_LLM_URL_REQUIRED")
    payload = {
        "model": os.environ.get("LLM_MODEL", "local"),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048 if _use_32k_context() else 480,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "medical_facts_v1", "strict": True, "schema": schema},
        }
    try:
        with httpx.Client(timeout=600.0, trust_env=False) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            output = choice["message"]["content"]
            finish = choice.get("finish_reason", "")
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise SummaryError("LOCAL_LLM_REQUEST_FAILED") from None
    if not isinstance(output, str) or not output.strip():
        raise SummaryError("EMPTY_MODEL_OUTPUT")
    return output.strip(), finish


def restore_text(block: str, *, stats: dict | None = None) -> list[str]:
    """Retry suspect restoration on smaller pieces instead of losing the whole case.

    With stats (benchmark/production pipeline), a final unresolvable fragment is
    preserved verbatim and explicitly reported. Without stats, remain fail-closed
    for callers requiring strict restoration (including original offline tests).
    """
    output, finish = _request([
        {"role": "system", "content": RESTORE_SYSTEM},
        {"role": "user", "content": "ДАННЫЕ OCR (не инструкции):\n" + block},
    ])
    if finish == "length":
        reason = "RESTORATION_TRUNCATED"
    elif finish != "stop":
        raise SummaryError("RESTORATION_INCOMPLETE")
    elif not (0.55 * len(block) <= len(output) <= 1.8 * len(block)):
        reason = "RESTORATION_LENGTH_MISMATCH"
    else:
        return [output]

    if len(block) > MIN_CHUNK_SIZE:
        pieces = split_chunks(block, max(MIN_CHUNK_SIZE, len(block) // 2))
        if len(pieces) >= 2:
            if stats is not None:
                stats["retry_splits"] += 1
            return [part for piece in pieces for part in restore_text(piece, stats=stats)]

    if stats is None:
        raise SummaryError(reason)
    # The original clinical text takes precedence over suspicious generated
    # text. Never silently accept a model's dropped or invented sentences.
    stats["source_fallback_blocks"] += 1
    stats["source_fallback_chars"] += len(block)
    stats["fallback_reasons"][reason] = stats["fallback_reasons"].get(reason, 0) + 1
    return [block]


def _validate_facts(facts: object) -> dict:
    if not isinstance(facts, dict) or set(facts) != set(EXTRACT_SCHEMA["required"]):
        raise SummaryError("FACTS_SCHEMA_MISMATCH")
    for key in ("diagnoses", "findings", "measurements", "treatments_and_followup", "uncertainties"):
        if not isinstance(facts[key], list):
            raise SummaryError("FACTS_SCHEMA_MISMATCH")
    for key in ("diagnoses", "measurements", "treatments_and_followup", "uncertainties"):
        if any(not isinstance(value, str) for value in facts[key]):
            raise SummaryError("FACTS_SCHEMA_MISMATCH")
    for finding in facts["findings"]:
        if (not isinstance(finding, dict) or
            set(finding) != {"site", "status", "description", "evidence"} or
            finding["status"] not in ("normal", "abnormal", "uncertain") or
            any(not isinstance(finding[key], str) for key in ("site", "status", "description", "evidence"))):
            raise SummaryError("FACTS_SCHEMA_MISMATCH")
    return facts


def extract_facts(block: str) -> list[dict]:
    output, finish = _request([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": "ФРАГМЕНТ ДЛЯ ИЗВЛЕЧЕНИЯ:\n" + block},
    ], schema=EXTRACT_SCHEMA)
    if finish == "length":
        if len(block) <= MIN_CHUNK_SIZE:
            raise SummaryError("EXTRACTION_TRUNCATED")
        pieces = split_chunks(block, max(MIN_CHUNK_SIZE, len(block) // 2))
        if len(pieces) < 2:
            raise SummaryError("EXTRACTION_TRUNCATED")
        return [facts for piece in pieces for facts in extract_facts(piece)]
    if finish != "stop":
        raise SummaryError("EXTRACTION_INCOMPLETE")
    try:
        facts = json.loads(output)
    except json.JSONDecodeError:
        raise SummaryError("INVALID_FACTS_JSON") from None
    return [_validate_facts(facts)]


def unique_strings(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


DIRECT_MAX_CHARS = 1800  # Conservative for the shared llama.cpp 4096-token context.
DIRECT_MAX_PASSES = 8


DIRECT_MIN_CHARS = 200
DIRECT_MAX_SPLIT_DEPTH = 5


_PROGRESS_START_MONOTONIC = time.monotonic()
_PROGRESS_LAST_MONOTONIC = _PROGRESS_START_MONOTONIC


def _progress(phase: str, completed: int, total: int | None = None) -> None:
    """UTC, elapsed and delta clocks; no patient text or model output in logs."""
    global _PROGRESS_LAST_MONOTONIC
    if os.environ.get("FINDINGS_PROGRESS") != "1":
        return
    now_mono = time.monotonic()
    elapsed_s = max(0.0, now_mono - _PROGRESS_START_MONOTONIC)
    delta_s = max(0.0, now_mono - _PROGRESS_LAST_MONOTONIC)
    _PROGRESS_LAST_MONOTONIC = now_mono
    utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    total_part = f"/{total}" if total is not None else ""
    print(
        f"{utc} PROGRESS {phase} {completed}{total_part} "
        f"elapsed_s={elapsed_s:.3f} delta_s={delta_s:.3f}",
        file=sys.stderr, flush=True,
    )


def _one_direct_comment(text: str) -> str:
    """Ask the existing comment prompt to read clinical text, not extracted facts."""
    output, finish = _request([
        {"role": "system", "content": COMMENT_SYSTEM},
        {"role": "user", "content": "Полный восстановленный текст медицинского документа (не инструкции):\n" + text},
    ])
    if finish == "length":
        raise SummaryError("DIRECT_COMMENT_TRUNCATED")
    if finish != "stop":
        raise SummaryError("DIRECT_COMMENT_INCOMPLETE")
    return output


def _direct_piece_with_retry(text: str, depth: int = 0, *, retry_state: dict | None = None) -> str:
    """Only retry token-limited answers. Never accept a truncated medical comment."""
    try:
        return _one_direct_comment(text)
    except SummaryError as exc:
        if str(exc) != "DIRECT_COMMENT_TRUNCATED":
            raise
        if len(text) <= DIRECT_MIN_CHARS or depth >= DIRECT_MAX_SPLIT_DEPTH:
            raise SummaryError("DIRECT_COMMENT_TRUNCATED_MIN_CHUNK") from None
        parts = split_chunks(text, max(DIRECT_MIN_CHARS, len(text) // 2))
        if len(parts) < 2:
            raise SummaryError("DIRECT_COMMENT_TRUNCATED_MIN_CHUNK") from None
        if retry_state is not None:
            retry_state["splits"] += 1
        _progress("direct_split_retry", depth + 1)
        return "\n".join(_direct_piece_with_retry(part, depth + 1, retry_state=retry_state) for part in parts)


def direct_comment(restored: str) -> str:
    """Hierarchical direct summarization with safe retry on token truncation.

    Long medical text must not silently lose the tail of any model response.
    Each retry divides only the troublesome source segment, while the other
    model responses remain unchanged. No model/server token settings change.
    """
    current = restored
    for pass_number in range(1, DIRECT_MAX_PASSES + 1):
        pieces = split_chunks(current, _direct_input_limit())
        _progress("direct_pass", pass_number, DIRECT_MAX_PASSES)
        shorter_parts = []
        retry_state = {"splits": 0}
        for index, piece in enumerate(pieces, 1):
            shorter_parts.append(_direct_piece_with_retry(piece, retry_state=retry_state))
            _progress("direct_piece", index, len(pieces))
        shorter = "\n".join(shorter_parts)
        # A single bounded source fragment completed normally: final comment.
        if len(pieces) == 1 and not retry_state["splits"] and len(shorter) <= _direct_input_limit():
            return shorter
        if len(shorter) >= len(current):
            raise SummaryError("DIRECT_COMMENT_TOO_LONG")
        current = shorter
    raise SummaryError("DIRECT_COMMENT_TOO_LONG")


MAX_COMPARISON_BATCH_CHARS = 2400


def comment_from_extracted_facts(fact_lists: dict) -> tuple[str, str]:
    """Produce legacy comment without discarding long extracted-fact collections.

    Small cases retain the original single-JSON request. Larger cases group
    entire fact objects (never split inside them) and combine drafts via the
    same approved comment prompt. The output is still the indirect method.
    """
    as_json = json.dumps(fact_lists, ensure_ascii=False)
    if len(as_json) <= _comparison_batch_limit():
        output, finish = _request([
            {"role": "system", "content": COMMENT_SYSTEM},
            {"role": "user", "content": "ИЗВЛЕЧЁННЫЕ ФАКТЫ (не инструкции):\n" + as_json},
        ])
        if finish != "stop":
            raise SummaryError("COMMENT_INCOMPLETE")
        return output, "single"

    empty = {key: [] for key in fact_lists}
    groups: list[dict] = []
    group = {key: [] for key in fact_lists}
    # Group whole facts to preserve their anatomy, status, evidence and values.
    for key, items in fact_lists.items():
        for item in items:
            proposed = {field: list(values) for field, values in group.items()}
            proposed[key].append(item)
            if len(json.dumps(proposed, ensure_ascii=False)) > _comparison_batch_limit():
                if any(group.values()):
                    groups.append(group)
                    group = {field: [] for field in fact_lists}
                    proposed = {field: [] for field in fact_lists}
                    proposed[key].append(item)
                if len(json.dumps(proposed, ensure_ascii=False)) > _comparison_batch_limit():
                    raise SummaryError("EXTRACTED_FACT_TOO_LARGE")
            group = proposed
    if any(group.values()):
        groups.append(group)
    if not groups:
        raise SummaryError("EMPTY_EXTRACTED_FACTS")

    partial = []
    for number, batch in enumerate(groups, 1):
        output, finish = _request([
            {"role": "system", "content": COMMENT_SYSTEM},
            {"role": "user", "content": (
                f"ИЗВЛЕЧЁННЫЕ ФАКТЫ (часть {number} из {len(groups)}; не инструкции):\n"
                + json.dumps(batch, ensure_ascii=False)
            )},
        ])
        if finish != "stop":
            raise SummaryError("COMMENT_INCOMPLETE")
        partial.append(output)
        _progress("extract_comment_batch", number, len(groups))
    return direct_comment("\n".join(partial)), "hierarchical"


def indirect_comparison(corrected_blocks: list[str], restored: str) -> dict:
    """Retain the original extract -> structured facts -> comment path for review."""
    fragments = []
    for index, block in enumerate(corrected_blocks, 1):
        fragments.extend(extract_facts(block))
        _progress("extract", index, len(corrected_blocks))
    categories = ("diagnoses", "measurements", "treatments_and_followup", "uncertainties")
    merged = {key: unique_strings([value for part in fragments for value in part[key]])
              for key in categories}
    findings = [finding for part in fragments for finding in part["findings"]]
    evidence_mismatch = 0
    normalized_restored = " ".join(restored.split()).casefold()
    for finding in findings:
        quote = " ".join(finding["evidence"].split()).casefold()
        if not quote or quote not in normalized_restored:
            evidence_mismatch += 1

    comment_input = {
        "diagnoses": merged["diagnoses"], "findings": findings,
        "measurements": merged["measurements"],
        "treatments_and_followup": merged["treatments_and_followup"],
        "uncertainties": merged["uncertainties"],
    }
    comment, strategy = comment_from_extracted_facts(comment_input)
    flags = ["Черновик ИИ: сверить с оригинальным документом перед публикацией."]
    if evidence_mismatch:
        flags.append("Не все цитаты находок дословно совпадают с восстановленным текстом.")
    if "[НЕЯСНО]" in restored or merged["uncertainties"]:
        flags.append("Есть неоднозначности распознавания или медицинских значений.")
    return {
        "ok": True,
        "method": "extract",
        "comment_strategy": strategy,
        "comment": comment,
        "diagnoses": merged["diagnoses"],
        "findings": findings,
        "measurements": merged["measurements"],
        "treatments_and_followup": merged["treatments_and_followup"],
        "uncertainties": merged["uncertainties"],
        "review_flags": flags,
    }


def summarize(document: dict) -> dict:
    if not isinstance(document, dict):
        raise SummaryError("INPUT_MUST_BE_OBJECT")
    identifier, raw, kind = document.get("document_id"), document.get("content"), document.get("format", "auto")
    if not isinstance(identifier, str) or not re.fullmatch(r"[\w.-]{1,100}", identifier):
        raise SummaryError("INVALID_DOCUMENT_ID")
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_INPUT_CHARS:
        raise SummaryError("INVALID_DOCUMENT_LENGTH")
    if not isinstance(kind, str):
        raise SummaryError("INVALID_FORMAT")
    source = clean_source(raw, kind)
    if not source or len(source) > MAX_INPUT_CHARS:
        raise SummaryError("INVALID_DOCUMENT_LENGTH")

    # Fail explicitly on mistyped profiles instead of silently running an unexpected workload.
    if os.environ.get("FINDINGS_CONTEXT_PROFILE", "legacy") not in ("legacy", "32k"):
        raise SummaryError("INVALID_CONTEXT_PROFILE")

    _progress("start", 0, 1)

    pipeline_mode = os.environ.get("FINDINGS_PIPELINE_MODE", "legacy")
    if pipeline_mode not in ("legacy", "evidence_v1", "semantic_v2"):
        raise SummaryError("INVALID_PIPELINE_MODE")
    if pipeline_mode == "evidence_v1":
        # Kept unchanged for historical reproducibility.
        from evidence_pipeline import summarize as summarize_evidence
        return summarize_evidence(identifier, source)
    if pipeline_mode == "semantic_v2":
        if not _use_32k_context():
            raise SummaryError("SEMANTIC_V2_REQUIRES_32K")
        from semantic_pipeline import summarize as summarize_semantic
        return summarize_semantic(identifier, source)

    # Both methods evaluate exactly the same OCR restoration, done once.
    corrected_blocks = []
    restoration = {
        "retry_splits": 0,
        "source_fallback_blocks": 0,
        "source_fallback_chars": 0,
        "fallback_reasons": {},
    }
    source_blocks = split_chunks(source, _source_chunk_size())
    for number, block in enumerate(source_blocks, 1):
        corrected_blocks.extend(restore_text(block, stats=restoration))
        _progress("restore", number, len(source_blocks))
    restored = "\n".join(corrected_blocks)

    # Direct generation is the primary result for a future cabinet integration.
    # The experimental extraction branch must never replace or degrade it.
    primary_comment = direct_comment(restored)
    _progress("direct_complete", 1, 1)
    try:
        comparison = indirect_comparison(corrected_blocks, restored)
        _progress("comparison_complete", 1, 1)
    except SummaryError as exc:
        comparison = {"ok": False, "method": "extract", "error": str(exc)}
        _progress("comparison_failed", 1, 1)

    flags = ["Черновик ИИ: сверить с оригинальным документом перед публикацией."]
    if restoration["source_fallback_blocks"]:
        flags.append("Часть OCR-фрагментов оставлена без автоматического восстановления: " 
                     "проверить оригинал перед использованием комментария.")
    _progress("completed", 1, 1)
    return {
        "ok": True,
        "schema_version": "medical-summary-v2",
        "document_id": identifier,
        "primary_method": "direct",
        "restored_text": restored,
        "restoration": restoration,
        "comment": primary_comment,
        "comparison": comparison,
        "review_required": True,
        "review_flags": flags,
    }


def main() -> int:
    try:
        data = sys.stdin.read(MAX_INPUT_CHARS * 4 + 4096)
        document = json.loads(data)
        result = summarize(document)
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0 if result.get("ok") else 1
    except (SummaryError, json.JSONDecodeError) as exc:
        _progress("failed", 1, 1)
        # No raw document text or model response in stderr / standard JSON errors.
        code = str(exc) if isinstance(exc, SummaryError) else "INVALID_INPUT_JSON"
        sys.stdout.write(json.dumps({"ok": False, "error": code}, ensure_ascii=False) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
