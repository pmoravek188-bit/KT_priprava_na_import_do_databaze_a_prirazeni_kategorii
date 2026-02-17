"""
AI kategorizace (OpenAI Responses API)

Cíl: minimalizovat "halucinace" tím, že model dostane uzavřený seznam kategorií

Prompty jsou schválně na jednom místě, aby se daly ladit bez zásahu do zbytku kódu.
"""

import json
import time
import random
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple



# PROMPT SPACE - edituj tady


# MINIMÁLNÍ SYSTEM PROMPT - bez ručních pravidel
SYSTEM_PROMPT = (
    "Jsi klasifikátor produktů do kategorií podle přesného číselníku. "
    "Dostaneš název produktu a seznam kategorií (id, název, popis). "
    "Použij pouze tyto kategorie a vyber nejvhodnější. "
    "Pokud je název v cizím jazyce nebo obsahuje překlepy/zkratky, pokus se odhadnout význam. "
    "MUSÍŠ vybrat jednu nejbližší kategorii podle názvu a popisu; "
    "nepoužívej žádné kategorie mimo seznam. "
    "Řiď se striktně názvem produktu a popisem kategorií. "
    "Postup: (1) normalizuj význam názvu (jazyk, překlepy), "
    "odhadni typ produktu (nápoj / jídlo / ingredience / surovina) a jeho stav "
    "(např. čerstvé, mražené, sušené, konzervované, hotové) pouze z názvu. "
    "Při interpretaci názvu se soustřeď na hlavní surovinu/jádro produktu; "
    "dochucení, omáčky, máslo, koření nebo přílohy ber jako doplňkové informace. "
    "Pokud je část názvu nejasná nebo zkomolená, neopírej klasifikaci o tuto část; "
    "spolehni se na ostatní jednoznačné složky názvu. "
    "Pokud je v názvu uveden způsob úpravy (např. uzené, pečené, grilované, smažené), "
    "drž se tohoto procesu a nepřekládaj ho na jiný. "
    "Sušené používej jen pokud je výslovně uvedeno \"sušené\" nebo ekvivalent (např. jerky). "
    "Uzené/zaúzené znamená uzené (smoked), není to sušené. "
    "Pokud název obsahuje konkrétní termín zpracování, preferuj kategorie, "
    "jejichž popis tento termín výslovně zmiňuje. "
    "Pokud název obsahuje uzené/zaúzené a neobsahuje sušené/jerky, "
    "nevybírej kategorie, jejichž název obsahuje slovo \"sušené\". "
    "(2) porovnej s popisy kategorií a vyber tu, která nejpřesněji odpovídá typu i stavu. "
    "Preferuj doslovnou shodu s popisem před volnými asociacemi "
    "a preferuj nejkonkrétnější kategorii podle popisu. "
    "(3) Pokud je název krátký a bez kontextu, neodhaduj zbytečné detaily; "
    "vyber nejbližší obecnou kategorii, která popisuje takový typ produktu. "
    "Vyhýbej se kategoriím, jejichž popis vyžaduje kontext, který v názvu chybí. "
    "Confidence je tvoje odhadovaná pravděpodobnost správnosti v rozsahu 0–1. "
    "Odpověz ve formátu JSON podle zadaného schématu a nepřidávej žádný další text."
)

USER_PROMPT_TEMPLATE = """\
PRODUKT K ZAŘAZENÍ:
Název: {product_name}

ČÍSELNÍK POVOLENÝCH KATEGORIÍ:
{categories}
"""

BATCH_USER_PROMPT_TEMPLATE = """\
PRODUKTY K ZAŘAZENÍ (každý má index):
{product_list}

ČÍSELNÍK POVOLENÝCH KATEGORIÍ:
{categories}

VÝSTUP:
Musíš vrátit výsledky pro VŠECHNY produkty (celkem {total_count}),
s indexy 1..{total_count} přesně jednou. Nic nevynechávej.
Každý výsledek musí obsahovat `index` (stejný jako ve vstupu),
`id_kategorie` a `confidence`.
"""


def _format_categories(categories: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for c in categories:
        cid = c.get("id_kategorie")
        name = (c.get("nazev") or "").strip()
        desc = (c.get("popis") or "").strip()
        if cid is None:
            continue
        if desc:
            lines.append(f"- {cid}: {name} | {desc}")
        else:
            lines.append(f"- {cid}: {name}")
    return "\n".join(lines)


def _call_openai_responses(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
    schema_name: str,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,
    max_output_tokens: int = 80,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    timeout_s: int = 30,
) -> Tuple[str, Dict[str, int]]:
    """
    Minimalní klient přes HTTP bez závislostí (nevyžaduje instalaci openai sdk).
    """
    url = "https://api.openai.com/v1/responses"
    text_payload: Dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "schema": json_schema,
            "strict": True,
        }
    }
    if text_verbosity:
        text_payload["verbosity"] = text_verbosity

    payload = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "text": text_payload,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    payload_no_cache = None
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        payload["prompt_cache_retention"] = prompt_cache_retention
    if prompt_cache_key or prompt_cache_retention:
        payload_no_cache = dict(payload)
        payload_no_cache.pop("prompt_cache_key", None)
        payload_no_cache.pop("prompt_cache_retention", None)

    def _make_request(body: Dict[str, Any]) -> urllib.request.Request:
        data = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

    current_payload = payload
    tried_no_cache = False
    req = _make_request(current_payload)
    # Retry logika pro 429/5xx a dočasné síťové chyby
    max_retries = 6
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            if (
                parsed.get("status") == "incomplete"
                and (parsed.get("incomplete_details", {}) or {}).get("reason") == "max_output_tokens"
                and attempt < max_retries - 1
            ):
                current_payload["max_output_tokens"] = min(
                    4000, int(current_payload.get("max_output_tokens", max_output_tokens) * 2)
                )
                if "reasoning" in current_payload:
                    current_payload["reasoning"] = {"effort": "minimal"}
                req = _make_request(current_payload)
                time.sleep(_retry_wait_seconds(attempt, None))
                continue
            return _extract_output_text(parsed), _extract_usage(parsed)
        except urllib.error.HTTPError as e:
            code = e.code
            if (
                code == 400
                and payload_no_cache is not None
                and not tried_no_cache
            ):
                tried_no_cache = True
                current_payload = payload_no_cache
                req = _make_request(current_payload)
                continue
            if attempt < max_retries - 1 and (code == 429 or code >= 500):
                time.sleep(_retry_wait_seconds(attempt, e))
                continue
            # Jiné chyby propaguj výš
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            raise RuntimeError(f"OpenAI HTTP {e.code}: {body or e.reason}") from e
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(_retry_wait_seconds(attempt, None))
                continue
            raise


def _extract_output_text(response_json: Dict[str, Any]) -> str:
    direct_text = response_json.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()
    output = response_json.get("output", []) or []
    parts: List[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            ctype = content.get("type")
            if ctype in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    return "".join(parts).strip()


def _extract_usage(response_json: Dict[str, Any]) -> Dict[str, int]:
    usage = response_json.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    input_details = usage.get("input_tokens_details", {}) or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
    }


def _retry_wait_seconds(attempt: int, err: Optional[urllib.error.HTTPError]) -> float:
    retry_after = None
    if err is not None:
        try:
            retry_after = err.headers.get("Retry-After")
        except Exception:
            retry_after = None
    base = min(8.0, 1.0 * (2 ** attempt))
    if retry_after:
        try:
            base = max(base, float(retry_after))
        except Exception:
            pass
    # Malý jitter, aby se požadavky nerozjely ve vlnách
    return base + random.uniform(0.0, 0.5)


def _build_messages(product_name: str, categories_text: str) -> List[Dict[str, Any]]:
    system_text = SYSTEM_PROMPT
    user_text = USER_PROMPT_TEMPLATE.format(
        product_name=product_name.strip(),
        categories=categories_text,
    )
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        },
    ]


def _build_messages_batch(product_names: Sequence[str], categories_text: str) -> List[Dict[str, Any]]:
    product_list = "\n".join(f"- {i+1} | {n}" for i, n in enumerate(product_names))
    user_text = BATCH_USER_PROMPT_TEMPLATE.format(
        product_list=product_list,
        categories=categories_text,
        total_count=len(product_names),
    )
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        },
    ]


def predict_category(
    *,
    product_name: str,
    categories: Sequence[Dict[str, Any]],
    api_key: str,
    model: str = "5.1 mini",
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
) -> Tuple[Optional[int], Optional[float], Dict[str, int]]:
    """
    Vrátí id_kategorie nebo None (když model vrátí null / nepůjde validovat).
    """
    allowed_ids = {int(c["id_kategorie"]) for c in categories if c.get("id_kategorie") is not None}

    categories_text = _format_categories(categories)
    messages = _build_messages(product_name, categories_text)
    single_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id_kategorie": {"type": "integer"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["id_kategorie", "confidence"],
    }

    content, usage = _call_openai_responses(
        api_key=api_key,
        model=model,
        messages=messages,
        json_schema=single_schema,
        schema_name="category_result",
        temperature=None,
        reasoning_effort="minimal",
        text_verbosity="low",
        max_output_tokens=200,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )
    raw = (content or "").strip()
    if raw == "":
        raise ValueError("Empty model output")
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError("Invalid JSON from model") from e
    cid = data.get("id_kategorie")
    conf = data.get("confidence")
    if cid is None:
        raise ValueError("Missing id_kategorie")
    if isinstance(cid, bool):
        raise ValueError("Invalid id_kategorie type")
    if isinstance(cid, str):
        if cid.strip().isdigit():
            cid = int(cid.strip())
        else:
            raise ValueError("Non-numeric id_kategorie")
    if not isinstance(cid, int):
        raise ValueError("Invalid id_kategorie type")
    if cid not in allowed_ids:
        raise ValueError("id_kategorie not in allowed list")
    return cid, _parse_confidence(conf), usage


def predict_categories_batch(
    *,
    product_names: Sequence[str],
    categories: Sequence[Dict[str, Any]],
    api_key: str,
    model: str = "5.1 mini",
    prompt_cache_key: Optional[str] = None,
    prompt_cache_retention: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[Dict[str, Tuple[Optional[int], Optional[float]]], Dict[str, int]]:
    allowed_ids = {int(c["id_kategorie"]) for c in categories if c.get("id_kategorie") is not None}
    categories_text = _format_categories(categories)
    messages = _build_messages_batch(product_names, categories_text)

    batch_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "id_kategorie": {"type": "integer"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["index", "id_kategorie", "confidence"],
                },
            }
        },
        "required": ["results"],
    }

    if max_output_tokens is None:
        max_output_tokens = max(500, len(product_names) * 120)

    content, usage = _call_openai_responses(
        api_key=api_key,
        model=model,
        messages=messages,
        json_schema=batch_schema,
        schema_name="category_results_batch",
        temperature=None,
        reasoning_effort="minimal",
        text_verbosity="low",
        max_output_tokens=max_output_tokens,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
    )

    raw = (content or "").strip()
    if raw == "":
        raise ValueError("Empty model output")
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError("Invalid JSON from model") from e

    results = data.get("results") or []
    out: Dict[str, Tuple[Optional[int], Optional[float]]] = {}
    total = len(product_names)
    for item in results:
        idx = item.get("index")
        try:
            idx = int(idx)
        except Exception:
            continue
        if idx < 1 or idx > total:
            continue
        name = product_names[idx - 1]
        cid = item.get("id_kategorie")
        conf = item.get("confidence")
        if isinstance(cid, str) and cid.strip().isdigit():
            cid = int(cid.strip())
        if not isinstance(cid, int) or cid not in allowed_ids:
            out[name] = (None, _parse_confidence(conf))
            continue
        out[name] = (cid, _parse_confidence(conf))

    # ensure all inputs present
    for n in product_names:
        if n not in out:
            out[n] = (None, None)
    return out, usage


def predict_category_id(
    *,
    product_name: str,
    categories: Sequence[Dict[str, Any]],
    api_key: str,
    model: str = "5.1 mini",
) -> Optional[int]:
    cid, _, _ = predict_category(
        product_name=product_name,
        categories=categories,
        api_key=api_key,
        model=model,
    )
    return cid


def _parse_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        conf = float(value)
    except Exception:
        return None
    if conf < 0 or conf > 1:
        return None
    return conf
