#!/bin/bash
set -e

echo "== Git LFS Safety Setup =="

# 1. Ensure LFS is installed
echo "[1/4] Installing Git LFS (if not already installed)..."
git lfs install --skip-repo || git lfs install

# 2. Track all the file types we want LFS to handle
echo "[2/4] Tracking large file patterns with Git LFS..."
git lfs track "*.csv.gz"
git lfs track "*.csv"
git lfs track "*.npz"
git lfs track "*.png"
git lfs track "*.gz"
git lfs track "data/samples/workload_samples*"

# 3. Add .gitattributes if changed
echo "[3/4] Adding .gitattributes to Git..."
git add .gitattributes

# 4. Commit and push
echo "[4/4] Committing LFS tracking changes..."
git commit -m "Ensure Git LFS tracking for large data files" || \
  echo "No changes to commit."

echo ""
echo "If this was a repo reset or filter-repo cleanup:"
echo "  Run: git push origin main --force"
echo ""
echo "== Done =="
