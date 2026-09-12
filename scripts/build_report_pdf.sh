#!/usr/bin/env bash
# Build a PDF from a Markdown report: build_report_pdf.sh [report.md]
# The PDF lands next to the source. With no argument, builds
# report/target_realism_test_report.md.
#
# Engine: pandoc + xelatex, DejaVu fonts (broad Unicode coverage for the Greek
# letters, math symbols, and box-drawing used in the report).
#
# Glyph handling:
#   - ✅ -> ✓ and ⟂ -> ⊥ are preprocessed (the originals are missing from the
#     fonts / parse awkwardly).
#   - ✓ and ⚠ fall back to DejaVu Sans Mono via a newunicodechar header.
#
# YAML metadata parsing is disabled (markdown-yaml_metadata_block) so the Part I
# header rule and the Part II divider (`---`) render as horizontal rules rather
# than being mistaken for a YAML block. Section numbers are authored by hand in
# the headings, so --number-sections is intentionally NOT passed.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:-report/target_realism_test_report.md}
OUT="${SRC%.md}.pdf"
PP="$(mktemp /tmp/tr_report_XXXXXX.md)"
HDR="$(mktemp /tmp/tr_header_XXXXXX.tex)"
trap 'rm -f "$PP" "$HDR"' EXIT

sed 's/✅/✓/g; s/⟂/⊥/g' "$SRC" > "$PP"

cat > "$HDR" <<'EOF'
\usepackage{newunicodechar}
\newunicodechar{✓}{{\fontspec{DejaVu Sans Mono}✓}}
\newunicodechar{⚠}{{\fontspec{DejaVu Sans Mono}⚠}}
EOF

pandoc "$PP" -o "$OUT" \
  --from=markdown-yaml_metadata_block \
  --pdf-engine=xelatex \
  --toc \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V colorlinks=true \
  --resource-path=report \
  -H "$HDR"

echo "wrote $OUT"
