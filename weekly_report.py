#!/usr/bin/env python3
"""Generate a weekly AI + Robotics newsletter from RSS feeds using OpenAI."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
from dateutil import parser as date_parser
from openai import OpenAI

SOURCES_FILE = Path("sources.json")
OUTPUT_FILE = Path("weekly_report.md")
DAYS_BACK = 7
DEFAULT_MODEL = "gpt-4o-mini"


def load_sources(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    feeds = data.get("feeds", [])
    if not feeds:
        raise ValueError("No feeds found in sources.json")
    return feeds


def entry_datetime(entry: Any) -> datetime | None:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if getattr(entry, "updated_parsed", None):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

    for field in ("published", "updated"):
        raw = getattr(entry, field, None)
        if raw:
            try:
                dt = date_parser.parse(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return None


def fetch_recent_items(feeds: list[dict[str, str]], days_back: int = DAYS_BACK) -> list[dict[str, str]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_back)
    items: list[dict[str, str]] = []

    for feed in feeds:
        name = feed["name"]
        url = feed["url"]
        parsed = feedparser.parse(url)

        for entry in parsed.entries:
            dt = entry_datetime(entry)
            if dt is None or dt < cutoff:
                continue

            items.append(
                {
                    "source": name,
                    "title": getattr(entry, "title", "(no title)"),
                    "link": getattr(entry, "link", ""),
                    "published": dt.strftime("%Y-%m-%d"),
                    "summary": getattr(entry, "summary", "").strip(),
                }
            )

    items.sort(key=lambda x: x["published"], reverse=True)
    return items


def build_prompt(items: list[dict[str, str]]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        lines.append(
            "\n".join(
                [
                    f"{i}. [{item['source']}] {item['title']}",
                    f"Date: {item['published']}",
                    f"Link: {item['link']}",
                    f"Summary: {item['summary'][:500]}",
                ]
            )
        )

    articles_blob = "\n\n".join(lines)
    return f"""
You are writing a concise weekly AI + Robotics newsletter in Markdown.

Requirements:
- Title: "AI + Robotics Weekly Report"
- Include period covered (last 7 days).
- Sections:
  1) Top AI Stories
  2) Top Robotics Stories
  3) Cross-over Trends (AI x Robotics)
  4) Notable Links (bullet list with markdown links)
- Keep it clear, factual, and skimmable.
- Only use the provided items.

Items:
{articles_blob}
""".strip()


def generate_markdown_report(items: list[dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    if not items:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            "# AI + Robotics Weekly Report\n\n"
            f"_Generated on {today}. No items found in the last 7 days from configured feeds._\n"
        )

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = build_prompt(items)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You produce high-quality technical newsletters in markdown.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text.strip() + "\n"


def main() -> None:
    feeds = load_sources(SOURCES_FILE)
    items = fetch_recent_items(feeds, days_back=DAYS_BACK)
    report = generate_markdown_report(items, model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE} with {len(items)} source items from last {DAYS_BACK} days.")


if __name__ == "__main__":
    main()
