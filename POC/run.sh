#!/usr/bin/env bash
# Launcher for core.py with KEY=VALUE overrides
# Default input and output directories
INPUT_DIR="input"
OUTPUT_DIR="output"

print_help() {
    cat <<EOF

Usage: $0 [dhash|phash|whash|reset|help] [image|video] [KEY=VALUE]...

Modes:
  dhash               Difference hash (fast, near-identical images)
  phash               Perceptual hash (DCT-based, robust to transforms)
  whash               Wavelet hash (texture/edge-sensitive)
  reset               Move files from output/ back into input/
  help                Show this help message

Override these Python defaults (flags):
  M_WORKERS=#         → --workers
  C_CHUNK=#           → --chunk-coef
  S_FRAMES=#          → --frames-to-sample
  S_HASH=#            → --hash-size
  S_THRESH=#          → --threshold
  D_RUN=true          → --dry-run
  (plus --graph if you add that feature)
EOF
}

# No args or help
if [ "$#" -eq 0 ] || [ "$1" = "help" ]; then
    print_help; exit 0
fi

# Reset workspace?
reset_output() {
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Resetting: moving files back to $INPUT_DIR/"
        find "$OUTPUT_DIR" -type f -exec mv -t "$INPUT_DIR" {} +
        rm -rf "$OUTPUT_DIR"
    else
        echo "Nothing to reset; $OUTPUT_DIR not found."
    fi
    exit 0
}
if [ "$1" = "reset" ]; then
    reset_output
fi

# Grab hash-type and media arguments
HASH_TYPE="$1"; MEDIA="$2"; shift 2
if [[ ! "$HASH_TYPE" =~ ^(dhash|phash|whash)$ ]]; then
    echo "Error: first arg must be dhash, phash, or whash"; print_help; exit 1
fi
if [[ ! "$MEDIA" =~ ^(image|video)$ ]]; then
    echo "Error: second arg must be image or video"; print_help; exit 1
fi

# Parse KEY=VALUE overrides
for kv in "$@"; do
    case "$kv" in
        M_WORKERS=*)    WORKERS="${kv#*=}" ;;
        C_CHUNK=*)      CHUNK_COEF="${kv#*=}" ;;
        S_FRAMES=*)     FRAMES="${kv#*=}" ;;
        S_HASH=*)       HASH_SIZE="${kv#*=}" ;;
        S_THRESH=*)     THRESHOLD="${kv#*=}" ;;
        D_RUN=*)        DRY_RUN="${kv#*=}" ;;
        *) echo "Unknown override: $kv"; print_help; exit 1 ;;
    esac
done

# Ensure input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: input directory '$INPUT_DIR' not found."; exit 1
fi

# Build the command
CMD=(uv run python scripts/mediahash.py
     --hash-type "$HASH_TYPE"
     --mode "$MEDIA"
     --input "$INPUT_DIR"
     --output "$OUTPUT_DIR"
)

[ -n "$WORKERS"    ] && CMD+=(--workers       "$WORKERS")
[ -n "$CHUNK_COEF" ] && CMD+=(--chunk-coef    "$CHUNK_COEF")
[ -n "$FRAMES"     ] && CMD+=(--frames-to-sample "$FRAMES")
[ -n "$HASH_SIZE"  ] && CMD+=(--hash-size     "$HASH_SIZE")
[ -n "$THRESHOLD"  ] && CMD+=(--threshold     "$THRESHOLD")
[ "$DRY_RUN" = "true" ] && CMD+=(--dry-run)

# (If you've implemented --graph:)
#[ "$GRAPH" = "true" ] && CMD+=(--graph)

echo "Executing: ${CMD[*]}"
"${CMD[@]}"
