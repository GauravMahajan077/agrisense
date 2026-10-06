# CELL 17 — crawler, quarantined
# NOT bulk-injected into training: web results are 20-40% mislabelled. Default role is a
# held-out field stress test. Wikimedia is tried first because it does not block datacentre
# IPs; DuckDuckGo usually does.
import requests

CR = WORK / "field_crawl"; CR.mkdir(parents=True, exist_ok=True)
prov = []
SESS = requests.Session()
# Wikimedia's UA policy asks for a descriptive agent with contact info.
SESS.headers.update({"User-Agent": "AgrisenseResearch/1.0 (academic rice-disease model; "
                                   "contact: gau.mah077@gmail.com)"})

def _fetch_bytes(url, max_bytes):
    # Stream the body and stop at max_bytes: a 20 MB "photo" is never useful, and reading it
    # whole wastes RAM and wall time on a crawl that is already time-boxed.
    with SESS.get(url, timeout=15, stream=True) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(1 << 16):
            buf.write(chunk)
            if buf.tell() > max_bytes:
                raise ValueError(f"body exceeds {max_bytes} bytes")
        return buf.getvalue()

def wiki_urls(q, n):
    r = SESS.get("https://commons.wikimedia.org/w/api.php", timeout=20, params={
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {q}", "gsrnamespace": "6", "gsrlimit": str(n*2),
        "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "640"})
    r.raise_for_status()
    for pg in (r.json().get("query", {}).get("pages", {}) or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("thumburl"):
            lic = (ii.get("extmetadata") or {}).get("LicenseShortName", {}).get("value", "?")
            yield ii["thumburl"], lic

def ddg_urls(q, n):
    try:
        from ddgs import DDGS
    except ImportError:
        try: from duckduckgo_search import DDGS
        except ImportError: return
    try:
        with DDGS() as d:
            for r in d.images(q, max_results=n*2):
                yield r["image"], "web-search-license-unknown"
    except Exception as e:
        print(f"   ddg blocked (datacentre IP): {str(e)[:60]}")

def crawl():
    t0 = time.time()
    for cls, queries in CFG["crawl"]["queries"].items():
        d = CR / cls; d.mkdir(exist_ok=True); got = 0
        stop = False
        for q in queries:
            for pv in CFG["crawl"]["providers"]:
                if got >= CFG["crawl"]["per_class"] or time.time()-t0 > CFG["crawl"]["max_seconds"]:
                    stop = True; break
                gen = (wiki_urls(q, CFG["crawl"]["per_class"]) if pv == "wikimedia"
                       else ddg_urls(q, CFG["crawl"]["per_class"]))
                try:
                    for url, lic in gen:
                        if got >= CFG["crawl"]["per_class"]: break
                        try:
                            with Image.open(io.BytesIO(_fetch_bytes(url, CFG["crawl"]["max_bytes"]))) as im:
                                if min(im.size) < CFG["crawl"]["min_side"] or \
                                   im.format not in ("JPEG", "PNG"): continue
                                p = d / f"{got:03d}.jpg"
                                im.convert("RGB").save(p, quality=92)
                        except Exception:
                            continue
                        prov.append({"class": cls, "file": str(p), "url": url, "query": q,
                                     "license": lic, "role": CFG["crawl"]["role"]})
                        got += 1; time.sleep(CFG["crawl"]["sleep"])
                except Exception as e:
                    print(f"   {pv} '{q[:26]}': {str(e)[:60]}")
            if stop: break
        print(f"  {cls}: {got} images")
    pd.DataFrame(prov).to_csv(CR / "provenance.csv", index=False)
    print(f"crawl done in {time.time()-t0:.0f}s -> {CR}")

assert CFG["crawl"]["role"] == "stress_test", "finetune role was removed on purpose"
if CFG["crawl"]["enable"] and not CFG["smoke"]:
    crawl()
    rows = []
    for p in sorted(CR.rglob("*.jpg")):
        with Image.open(p) as im:
            # 0-255 floats: EfficientNet preprocesses internally, so no /255 here. PIL resize
            # takes a Resampling enum (np.float32 raises "Unknown resampling filter"), and a
            # PIL Image is not subscriptable — np.asarray must come before [None].
            x = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR),
                           dtype=np.float32)[None]
        pr = model.predict(x, verbose=0)[0]
        s = np.sort(pr)
        rows.append({"true": p.parent.name, "pred": CLASSES[int(pr.argmax())],
                     "conf": round(float(pr.max()), 3),
                     "margin": round(float(s[-1]-s[-2]), 3)})
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== FIELD STRESS TEST ===\n" + df.to_string(index=False))
        print(f"top-1 {float((df['true']==df['pred']).mean()):.1%} | "
              f"mean margin {df['margin'].mean():.3f}")
        # abstain rule: same threshold as the held-out eval (Cell 16.5)
        thr = CFG["abstain_threshold"]
        cov = float((df["conf"] >= thr).mean())
        acc_cov = (float((df.loc[df["conf"] >= thr, "true"] ==
                          df.loc[df["conf"] >= thr, "pred"]).mean()) if cov > 0 else float("nan"))
        print(f"abstain@{thr}: coverage {cov:.1%} | top-1 on covered {acc_cov:.1%}")
        df.to_csv(OUT / "field_stress_test.csv", index=False)
        print("Crawled labels are themselves noisy. Read the pattern, not the number.")
