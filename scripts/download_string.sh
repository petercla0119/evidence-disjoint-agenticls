#!/usr/bin/env bash
# Fetch the two STRING v12.0 human files the packet builder reads (85 MB; not vendored).
set -euo pipefail
cd "$(dirname "$0")/../data/external"
base=https://stringdb-downloads.org/download
while read -r sha dir file; do
  [ -f "$file" ] || curl -fL -o "$file" "$base/$dir/$file"
  echo "$sha  $file" | shasum -a 256 -c -
done <<'LIST'
144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c protein.info.v12.0 9606.protein.info.v12.0.txt.gz
3e22f32572211aa341d5b4bd08d30c32e693e294603202120936872f87719d4f protein.links.v12.0 9606.protein.links.v12.0.txt.gz
LIST
