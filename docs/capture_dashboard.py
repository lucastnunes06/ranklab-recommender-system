"""Captura da interface HTML/CSS/JS com respostas do backend Python real.

Não depende de servidor remoto nem de dados reais. O TestClient calcula as
respostas das configurações demonstradas; Playwright renderiza a interface
original usando essas respostas em vez de fazer chamadas de rede.

    pip install -r requirements-preview.txt
    python -m playwright install chromium
    python -m docs.capture_dashboard
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from playwright.sync_api import sync_playwright

from ranklab.api import app

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "ranklab" / "static"
client = TestClient(app)


def capture() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    # Somente o carregamento externo dos ativos é substituído na captura.
    html = html.replace('<link href="/static/styles.css" rel="stylesheet"/>', f"<style>{css}</style>")
    html = html.replace('<script defer="" src="/static/app.js"></script>', "")
    html = html.replace('<script src="/static/app.js" defer></script>', "")

    demo_responses = {}
    for cuisine, diversity in [
        ("Pizza", 0.5), ("Pizza", 0.0), ("Pizza", 0.5),
        ("Pizza", 1.0), ("Japonesa", 1.0),
    ]:
        params = {
            "cuisine": cuisine,
            "budget": 2,
            "rush": True,
            "vegetarian": False,
            "diversity": diversity,
            "max_eta": 80,
        }
        key = f"{cuisine}|{diversity:g}"
        result = client.get("/api/recommendations", params=params)
        result.raise_for_status()
        demo_responses[key] = result.json()

    metrics = client.get("/api/metrics").json()
    study = client.get("/api/diversity-study").json()
    data = json.dumps({"recommendations": demo_responses, "metrics": metrics, "study": study}, ensure_ascii=False)
    mock = f"""
      const FIXTURE = {data};
      window.fetch = async function(url) {{
        const resource = new URL(url, 'http://ranklab.local/');
        let payload;
        if (resource.pathname === '/api/metrics') payload = FIXTURE.metrics;
        else if (resource.pathname === '/api/diversity-study') payload = FIXTURE.study;
        else if (resource.pathname === '/api/recommendations') {{
          const cuisine = resource.searchParams.get('cuisine') || 'Pizza';
          const diversity = Number(resource.searchParams.get('diversity') || 0);
          payload = FIXTURE.recommendations[`${{cuisine}}|${{diversity}}`];
        }}
        if (!payload) return {{ok:false,status:404,json:async()=>({{detail:'Cenário não incluído na captura'}})}};
        return {{ok:true,status:200,json:async()=>payload}};
      }};
    """

    with sync_playwright() as playwright:
        browser_path = shutil.which("chromium") or shutil.which("chromium-browser")
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        if browser_path:
            kwargs["executable_path"] = browser_path
        browser = playwright.chromium.launch(**kwargs)
        page = browser.new_page(viewport={"width": 1420, "height": 950}, device_scale_factor=1)
        page.set_content(html, wait_until="domcontentloaded")
        page.add_script_tag(content=mock)
        page.add_script_tag(content=js)
        page.wait_for_function("document.querySelectorAll('.restaurant').length >= 5")
        page.wait_for_function("document.querySelectorAll('#studyRows tr').length === 5")
        page.screenshot(path=str(ROOT / "docs" / "dashboard.png"), full_page=True)

        frames: list[Image.Image] = []
        def frame() -> None:
            raw = page.screenshot(full_page=False)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            width = 1000
            frames.append(image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS))

        frame()
        for level in ("0", "0.5", "1"):
            page.locator("#diversity").evaluate(
                "(el, v) => {el.value=v; el.dispatchEvent(new Event('input', {bubbles:true}));}", level
            )
            page.click("#apply")
            page.wait_for_function("document.querySelectorAll('.restaurant').length >= 5")
            page.wait_for_timeout(200)
            frame()
        page.locator("#cuisine").select_option(label="Japonesa")
        page.click("#apply")
        page.wait_for_timeout(300)
        frame()

        frames[0].save(
            ROOT / "docs" / "recommendations-demo.gif", save_all=True,
            append_images=frames[1:],
            duration=[1400, 1100, 1100, 1100, 2000], loop=0, optimize=True,
        )
        browser.close()
    print("Captura e GIF produzidos com a interface real e respostas do backend em TestClient.")


if __name__ == "__main__":
    capture()
