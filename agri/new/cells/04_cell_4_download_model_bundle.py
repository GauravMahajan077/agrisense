# ============================================================
# CELL 4 — DOWNLOAD the trained model bundle
# ============================================================
# The trained bundle is /kaggle/working/agrisense_bundle.zip.
# NOTE: FileLink/URL links do NOT work on Kaggle (they 404). Use the
# Output tab or the kaggle CLI instead — see the instructions below.
import os, zipfile

bundle = "/kaggle/working/agrisense_bundle.zip"
assert os.path.exists(bundle), f"not found: {bundle} — run the full pipeline in THIS session first"
print(f"bundle: {bundle}  ({os.path.getsize(bundle)/1e6:.1f} MB)")
with zipfile.ZipFile(bundle) as z:
    for n in z.namelist():
        print(f"  {n:<28} {z.getinfo(n).file_size/1e6:6.2f} MB")

print("""
Download it (pick ONE):

  1) OUTPUT TAB (easiest): top-right of the notebook, click the Output icon
     (folder with an arrow). It shows /kaggle/working files. Click the
     Download button at the top of that panel -> you get a zip with
     agrisense_bundle.zip inside.

  2) SAVED VERSION: after you Save Version, open the notebook's page and the
     version's Output section lists the files with a Download button.

  3) KAGGLE CLI (run on YOUR local terminal, not here):
       kaggle kernels output <your-username>/<kernel-slug> -p ./agrisense
     This downloads the whole /kaggle/working output to ./agrisense.
""")