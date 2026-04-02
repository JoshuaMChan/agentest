from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytesseract
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from playwright.sync_api import Page, sync_playwright


ARTIFACTS_DIR = Path("artifacts")
MAX_AGENT_STEPS = 8
DOM_SCAN_LIMIT = 200


def _log(message: str) -> None:
    print(f"[run] {message}", flush=True)


@dataclass
class UiHit:
    source: str
    text: str
    bbox: tuple[float, float, float, float]
    screenshot_path: str


def _clip_to_viewport(
    x: float, y: float, width: float, height: float, viewport_w: float, viewport_h: float
) -> tuple[float, float, float, float] | None:
    x = max(0.0, x)
    y = max(0.0, y)
    width = min(width, viewport_w - x)
    height = min(height, viewport_h - y)
    if width <= 2 or height <= 2:
        return None
    return x, y, width, height


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _visible_text_preview(page: Page, max_chars: int = 3000) -> str:
    body_text = page.locator("body").inner_text(timeout=5000)
    body_text = re.sub(r"\s+", " ", body_text).strip()
    return body_text[:max_chars]


def _first_visible_selector(page: Page, selectors: list[str]) -> str | None:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.count() > 0 and locator.is_visible(timeout=600):
                return selector
        except Exception:  # noqa: BLE001
            continue
    return None


def _click_search_with_llm(page: Page, llm: ChatGoogleGenerativeAI, keyword: str) -> None:
    last_error: Exception | None = None
    for url in ("https://www.baidu.com", "http://www.baidu.com", "https://m.baidu.com"):
        try:
            _log(f"opening page: {url}")
            page.goto(url, wait_until="commit", timeout=60000)
            page.wait_for_timeout(1200)
            if "baidu.com" in page.url:
                _log(f"opened successfully: {page.url}")
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _log(f"open failed, trying next url: {exc}")
    else:
        raise RuntimeError(f"Unable to open Baidu from this network: {last_error}")

    fallback_actions: list[dict[str, Any]] = [
        {"action": "type", "selector": "#kw", "text": keyword},
        {"action": "click", "selector": "#su"},
        {"action": "wait", "ms": 1800},
        {"action": "done"},
    ]

    for step in range(MAX_AGENT_STEPS):
        _log(f"agent step {step + 1}/{MAX_AGENT_STEPS}")
        page_title = page.title()
        page_url = page.url
        preview = _visible_text_preview(page)
        prompt = (
            "You are controlling a browser robot.\n"
            "Goal: search keyword on Baidu and reach first result page.\n"
            "Return STRICT JSON only.\n"
            "Allowed actions:\n"
            '1) {"action":"type","selector":"#kw","text":"..."}\n'
            '2) {"action":"click","selector":"#su"}\n'
            '3) {"action":"wait","ms":1000}\n'
            '4) {"action":"done"}\n'
            "Rules:\n"
            "- Use only selectors above unless unavailable.\n"
            "- Prefer type then click.\n"
            "- done only when URL indicates search results page.\n\n"
            f"Keyword: {keyword}\n"
            f"Current URL: {page_url}\n"
            f"Current title: {page_title}\n"
            f"Visible text preview: {preview}\n"
        )
        try:
            response = llm.invoke(
                [SystemMessage(content="Act as deterministic browser planner."), HumanMessage(content=prompt)]
            )
            raw = response.content if isinstance(response.content, str) else str(response.content)
            json_str = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            action = json.loads(json_str)
        except Exception:  # noqa: BLE001
            action = fallback_actions[min(step, len(fallback_actions) - 1)]
            _log("LLM unavailable, using fallback action")

        action_name = action.get("action")
        _log(f"action: {action_name}")
        if action_name == "type":
            selector = action.get("selector", "#kw")
            text = action.get("text", keyword)
            type_selector = _first_visible_selector(
                page, [selector, "input#kw", "input[name='wd']", "textarea[name='wd']", "input.s_ipt"]
            )
            if type_selector:
                locator = page.locator(type_selector).first
                locator.click()
                locator.fill("")
                locator.type(text, delay=50)
            else:
                page.keyboard.type(text, delay=50)
        elif action_name == "click":
            selector = action.get("selector", "#su")
            click_selector = _first_visible_selector(
                page, [selector, "input#su", "button#su", "input[type='submit']", "button[type='submit']"]
            )
            if click_selector:
                page.locator(click_selector).first.click(timeout=5000)
            else:
                page.keyboard.press("Enter")
        elif action_name == "wait":
            ms = int(action.get("ms", 800))
            page.wait_for_timeout(max(300, min(ms, 3000)))
        elif action_name == "done":
            if "wd=" in page.url or "baidu.com/s" in page.url:
                _log("agent reached result page")
                return
            page.wait_for_timeout(800)
        else:
            page.wait_for_timeout(800)

    type_selector = _first_visible_selector(page, ["input#kw", "input[name='wd']", "textarea[name='wd']", "input.s_ipt"])
    click_selector = _first_visible_selector(page, ["input#su", "button#su", "input[type='submit']", "button[type='submit']"])
    if type_selector and click_selector:
        page.locator(type_selector).first.fill(keyword)
        page.locator(click_selector).first.click()
    page.wait_for_timeout(1800)


def _collect_dom_hits(page: Page, keyword: str, artifacts_dir: Path) -> list[UiHit]:
    _log("DOM_SCAN start: collecting DOM text hits")
    keyword_lc = keyword.lower()
    viewport = page.viewport_size or {"width": 1366, "height": 768}
    candidates: list[dict[str, Any]] = page.evaluate(
        """
        (maxItems) => {
          const out = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
          while (walker.nextNode()) {
            const el = walker.currentNode;
            const text = (el.innerText || "").trim();
            if (!text) continue;
            const st = window.getComputedStyle(el);
            if (st.visibility === "hidden" || st.display === "none") continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < 20 || rect.height < 12) continue;
            if (rect.bottom < 0 || rect.right < 0) continue;
            out.push({
              text: text,
              x: rect.x,
              y: rect.y,
              width: rect.width,
              height: rect.height
            });
            if (out.length >= maxItems) break;
          }
          return out;
        }
        """,
        DOM_SCAN_LIMIT,
    )

    hits: list[UiHit] = []
    hit_idx = 0
    for idx, item in enumerate(candidates):
        text = str(item.get("text", ""))
        matched = keyword_lc in text.lower()
        _log(f"DOM_SCAN #{idx + 1} matched={matched} text={text}")
        if not matched:
            continue
        raw_clip = _clip_to_viewport(
            float(item["x"]) - 6,
            float(item["y"]) - 6,
            float(item["width"]) + 12,
            float(item["height"]) + 12,
            float(viewport["width"]),
            float(viewport["height"]),
        )
        if not raw_clip:
            continue
        x, y, w, h = raw_clip
        hit_idx += 1
        shot_path = artifacts_dir / f"dom_hit_{hit_idx}.png"
        page.screenshot(path=str(shot_path), clip={"x": x, "y": y, "width": w, "height": h})
        hits.append(UiHit(source="dom", text=text, bbox=(x, y, w, h), screenshot_path=str(shot_path)))
        _log(f"DOM_HIT #{hit_idx} text={text}")
    if len(candidates) >= DOM_SCAN_LIMIT:
        _log(f"DOM_SCAN stop reason: reached limit={DOM_SCAN_LIMIT}")
    _log(f"DOM_SCAN end: scanned={len(candidates)} hits_saved={len(hits)}")
    return hits


def _collect_ocr_hits(page: Page, keyword: str, artifacts_dir: Path) -> list[UiHit]:
    _log("OCR_SCAN start: running OCR on page screenshot")
    viewport = page.viewport_size or {"width": 1366, "height": 768}
    full_path = artifacts_dir / "full_page.png"
    page.screenshot(path=str(full_path), full_page=False)
    image = Image.open(full_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    keyword_lc = keyword.lower()
    hits: list[UiHit] = []
    all_ocr_texts = [str(word).strip() for word in data.get("text", []) if str(word).strip()]
    full_ocr_text = " ".join(all_ocr_texts)
    ocr_hit_logged = False

    for i, word in enumerate(data.get("text", [])):
        text = (word or "").strip()
        if not text:
            continue
        if keyword_lc not in text.lower():
            continue
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        clip = _clip_to_viewport(
            x - 10, y - 10, w + 20, h + 20, float(viewport["width"]), float(viewport["height"])
        )
        if not clip:
            continue
        cx, cy, cw, ch = clip
        shot_path = artifacts_dir / f"ocr_hit_{len(hits) + 1}.png"
        page.screenshot(path=str(shot_path), clip={"x": cx, "y": cy, "width": cw, "height": ch})
        hits.append(UiHit(source="ocr", text=text, bbox=(cx, cy, cw, ch), screenshot_path=str(shot_path)))
        _log(f"OCR_HIT #{len(hits)} text={text}")
        if not ocr_hit_logged:
            _log(f"OCR_TEXT full={full_ocr_text}")
            ocr_hit_logged = True
    _log(f"OCR_SCAN end: words={len(all_ocr_texts)} hits_saved={len(hits)}")
    return hits


def _dedupe_hits(hits: list[UiHit]) -> list[UiHit]:
    deduped: list[UiHit] = []
    for hit in hits:
        duplicate = False
        for kept in deduped:
            if _bbox_iou(hit.bbox, kept.bbox) > 0.55:
                duplicate = True
                break
        if not duplicate:
            deduped.append(hit)
    return deduped


def run_validation() -> None:
    _log("starting validation")
    load_dotenv()
    keyword = "百度"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Put it in .env or environment variables.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"clearing old screenshots in: {ARTIFACTS_DIR}")
    for old_png in ARTIFACTS_DIR.glob("*.png"):
        old_png.unlink(missing_ok=True)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=0,
    )

    with sync_playwright() as p:
        _log("launching headless chromium")
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
        )
        page = context.new_page()

        _click_search_with_llm(page, llm, keyword)
        page.wait_for_timeout(1500)
        _log("saving result page screenshot")
        page.screenshot(path=str(ARTIFACTS_DIR / "result_page.png"), full_page=False)

        dom_hits = _collect_dom_hits(page, keyword, ARTIFACTS_DIR)
        ocr_hits = _collect_ocr_hits(page, keyword, ARTIFACTS_DIR)
        all_hits = _dedupe_hits(dom_hits + ocr_hits)
        _log(f"deduped total hits: {len(all_hits)}")

        context.close()
        browser.close()
        _log("browser closed")

    print(f"keyword: {keyword}")
    print(f"occurrence_count: {len(all_hits)}")
    print("artifacts:")
    for hit in all_hits:
        print(f"- [{hit.source}] {hit.screenshot_path} ({hit.text})")


if __name__ == "__main__":
    started = time.time()
    run_validation()
    print(f"done_in_seconds: {time.time() - started:.2f}")
