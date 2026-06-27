"""
shared_utils/wiki.py
Resolve English Wikipedia seed article titles into a target-language edition via
interlanguage links (langlinks). Lets the topic collector author only English seed
lists instead of per-language title lists.
"""

import json
import urllib.parse
import urllib.request
from typing import Optional

_USER_AGENT = "CultureRiddlesResearch/1.0 (research@example.com)"


def resolve_langlink(seed_en: str, target_wiki: str, timeout: int = 20) -> Optional[str]:
    """
    Return the title of `seed_en` (an en.wikipedia article) in the `target_wiki`
    edition via langlinks, or None if there is no link / lookup fails.

    `target_wiki` is a Wikipedia edition code, e.g. "ar", "arz", "zh-yue".
    """
    if not seed_en or not target_wiki:
        return None
    params = {
        "action": "query",
        "format": "json",
        "titles": seed_en,
        "prop": "langlinks",
        "lllang": target_wiki,
        "lllimit": "1",
        "redirects": "1",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception:
        return None

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        for ll in page.get("langlinks", []):
            title = ll.get("*")
            if title:
                return title
    return None
