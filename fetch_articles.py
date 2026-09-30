import pathlib
import trafilatura

out = pathlib.Path("data/articles")
out.mkdir(parents=True, exist_ok=True)

for url in pathlib.Path("urls.txt").read_text().split():
    text = trafilatura.extract(trafilatura.fetch_url(url))
    if text:
        name = url.rstrip("/").split("/")[-1][:60]
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        print("saved", name)
    else:
        print("failed", url)