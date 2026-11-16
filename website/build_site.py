"""
Assembles the website from src/site.html + site_data.json.

  python website/build_site.py
writes
  website/index.html        full HTML document (open locally or serve with GitHub Pages)
  website/dist/page.html    the same page as a body fragment, for hosts that add their own <head>
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "src", "site.html"), encoding="utf-8").read()
data = json.load(open(os.path.join(HERE, "site_data.json"), encoding="utf-8"))
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
src = src.replace("/*__SITE_DATA__*/null", payload)

head = src[src.index("<!--HEAD-->") + len("<!--HEAD-->"):src.index("<!--/HEAD-->")]
body = src[src.index("<!--/HEAD-->") + len("<!--/HEAD-->"):]

full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
    f.write(full)
os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
with open(os.path.join(HERE, "dist", "page.html"), "w", encoding="utf-8") as f:
    f.write(head + body)
print("index.html", round(len(full) / 1e3), "KB")
