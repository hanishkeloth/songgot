#!/bin/bash
# After the Hugging Face org rename imcapsule -> palette-lab: sweep references, rebuild, push, update Hub cards.
set -e; cd ~/Desktop/SONGGOT
OLD=imcapsule; NEW=palette-lab
for f in README.md paper/SONGGOT.md PLAN.md harness/teacher_gen.py site/build.py site/publish.py space/README.md space/app.py MODEL_CARD.md; do
  sed -i '' "s|$OLD/|$NEW/|g; s|hf.co/$OLD|hf.co/$NEW|g; s|huggingface.co/$OLD|huggingface.co/$NEW|g; s|org $OLD|org $NEW|g" "$f"
done
.venv/bin/python site/build.py > /dev/null
grep -rIl --exclude-dir=.venv --exclude-dir=vol --exclude-dir=tools --exclude-dir=.git --exclude-dir=logs --exclude=rename_org.sh "$OLD" . && { echo "leftover references above"; exit 1; } || true
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/docs/songgot.pdf" --virtual-time-budget=5000 "file://$PWD/docs/index.html" 2>/dev/null || true
git add -A && git -c user.name="Hanish Keloth" -c user.email="4217831+hanishkeloth@users.noreply.github.com" commit -q -m "Hugging Face org renamed to palette-lab" && git push -q origin main
sed -i '' "s|$OLD|$NEW|g" /tmp/songgot_static/index.html
.venv/bin/python - <<PY
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md", repo_id="$NEW/songgot", repo_type="model", commit_message="Org renamed to palette-lab")
api.upload_folder(repo_id="Hanish/songgot", repo_type="space", folder_path="space", commit_message="Org renamed to palette-lab")
api.upload_folder(repo_id="$NEW/songgot", repo_type="space", folder_path="/tmp/songgot_static", commit_message="Org renamed to palette-lab")
print("hub updated")
PY
echo "RENAME SWEEP DONE"
