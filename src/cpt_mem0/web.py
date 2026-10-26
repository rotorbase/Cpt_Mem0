"""The web for the assistant - a tool the model can call.

**The model does not have the internet.** It is a local file of
weights on a graphics card, and it knows what was in the training
data and nothing else. But the app can go online, and it can hand
the model what it finds.

**Two ways out.** A search without a key, and a page that is read.
The search is DuckDuckGo's instant answer and Wikipedia - both of
them answer without an account, and both of them are German when
the question is German. The page is read directly and the tags are
stripped, so nothing runs.

**The switch belongs to the user.** An app that keeps everything on
the machine and then quietly asks the internet is something else
than what the user was promised. So the access is off until it is
switched on, every call is written into the log, and the window
shows what was asked.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

#: How long a request may take.
TIMEOUT = 20

#: How much of a page the model gets. Enough for the facts, not
#: enough to fill the context.
PAGE_LIMIT = 6000

#: The user agent. Some sites answer a browser and not a script.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "Cpt-Mem0/0.4 (local assistant)")


def _get(url: str, params: dict[str, str] | None = None) -> str:
    """Read a URL and return the body."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT,
                       "Accept-Language": "de,en;q=0.8"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def _text_of(html: str) -> str:
    """The readable text of a page.

    **The tags go, the words stay.** No script runs, nothing is
    loaded - the page is read as it stands there.
    """
    without_scripts = re.sub(
        r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
        " ", html, flags=re.IGNORECASE | re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    import html as html_module

    text = html_module.unescape(without_tags)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def search(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search the web without an account.

    **Two sources.** DuckDuckGo knows the quick answers - a weather
    forecast, a definition, a package version. Wikipedia knows the
    rest, and it knows it in German.
    """
    query = (query or "").strip()
    if not query:
        return []

    results: list[dict[str, str]] = []

    # DuckDuckGo: the instant answer.
    try:
        body = _get("https://api.duckduckgo.com/",
                    {"q": query, "format": "json", "no_html": "1",
                     "no_redirect": "1", "skip_disambig": "1"})
        data = json.loads(body)
        abstract = str(data.get("AbstractText") or "").strip()
        if abstract:
            results.append({
                "title": str(data.get("Heading") or query),
                "url": str(data.get("AbstractURL") or ""),
                "text": abstract[:PAGE_LIMIT],
                "source": "DuckDuckGo",
            })
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": query,
                    "url": str(topic.get("FirstURL") or ""),
                    "text": str(topic["Text"])[:PAGE_LIMIT],
                    "source": "DuckDuckGo",
                })
    except Exception:  # noqa: BLE001
        pass

    # Wikipedia: the articles.
    try:
        body = _get("https://de.wikipedia.org/w/api.php", {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": str(limit),
        })
        data = json.loads(body)
        for item in (data.get("query", {}).get("search", []) or [])[:limit]:
            title = str(item.get("title") or "")
            results.append({
                "title": title,
                "url": f"https://de.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                "text": _text_of(str(item.get("snippet") or "")),
                "source": "Wikipedia",
            })
    except Exception:  # noqa: BLE001
        pass

    return results[:limit]


def fetch(url: str) -> dict[str, str]:
    """Read a page and give the text to the model."""
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": "no", "text": "Das ist keine Adresse."}
    try:
        body = _get(url)
    except urllib.error.HTTPError as error:
        return {"ok": "no", "text": f"Die Seite antwortete mit {error.code}."}
    except Exception as error:  # noqa: BLE001
        return {"ok": "no", "text": f"Die Seite war nicht zu lesen: {error}"}

    text = _text_of(body)
    found = re.search(r"<title[^>]*>(.*?)</title>", body,
                      re.IGNORECASE | re.DOTALL)
    return {
        "ok": "yes",
        "url": url,
        "title": found.group(1).strip() if found else "",
        "text": text[:PAGE_LIMIT],
    }


#: The tools of the assistant, in the shape the OpenAI-compatible
#: endpoint of llama.cpp expects.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for facts that are not in the model. "
                "Use it for things that changed recently, for prices, "
                "for versions, and for anything the user says is "
                "wrong."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Read a page and get its text. Use it after a search "
                "when the snippet is not enough."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The address of the page.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def run(name: str, arguments: dict[str, Any]) -> str:
    """Run one tool and return what the model gets to see."""
    if name == "web_search":
        query = str(arguments.get("query") or "")
        found = search(query)
        if not found:
            return "Keine Treffer."
        parts = []
        for entry in found:
            parts.append(f"{entry['title']} ({entry['source']})\n"
                         f"{entry['text']}\n{entry['url']}")
        return "\n\n".join(parts)[:PAGE_LIMIT]

    if name == "web_fetch":
        result = fetch(str(arguments.get("url") or ""))
        if result.get("ok") != "yes":
            return str(result.get("text") or "Die Seite war nicht zu lesen.")
        return f"{result.get('title', '')}\n{result.get('text', '')}"

    return f"Das Werkzeug {name} gibt es nicht."
