import json, sys
from pathlib import Path
d = Path(__file__).parent
items = json.loads((d / "items.json").read_text(encoding="utf-8"))
urls = json.loads((d / "urls.json").read_text(encoding="utf-8")) if (d / "urls.json").exists() else {}
tpl = (d / "page.tpl.html").read_text(encoding="utf-8")
html = tpl.replace("__ITEMS__", json.dumps(items, separators=(",", ":"))).replace("__IMG__", json.dumps(urls))
(d / "audit.html").write_text(html, encoding="utf-8")
print(len(html), "bytes,", len(urls), "image urls")
