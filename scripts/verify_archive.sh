#!/bin/bash
#
# Verify an unpacked dataset archive against its checksum manifest.
#
# Usage:
#   scripts/verify_archive.sh /path/to/hdac-reproduction-archive
#
# Run this before copying data/ and results/ into the repository. The archive is
# large, and a truncated transfer does not announce itself: the representation
# caches are positional, so a partial file scores the wrong molecules and returns
# entirely plausible numbers rather than an error.

set -euo pipefail

ARCHIVE_DIR="${1:-}"
if [ -z "$ARCHIVE_DIR" ]; then
    echo "Usage: scripts/verify_archive.sh /path/to/hdac-reproduction-archive" >&2
    exit 1
fi
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "Error: not a directory: $ARCHIVE_DIR" >&2
    exit 1
fi

MANIFEST="$ARCHIVE_DIR/MANIFEST.sha256"
if [ ! -f "$MANIFEST" ]; then
    echo "Error: no MANIFEST.sha256 in $ARCHIVE_DIR" >&2
    echo "  Unzip the archive first, and point this script at the folder it creates." >&2
    exit 1
fi

echo "Verifying $(wc -l < "$MANIFEST") files in $ARCHIVE_DIR"
echo "This reads every byte of a multi-gigabyte archive and takes a few minutes."

if ( cd "$ARCHIVE_DIR" && sha256sum -c MANIFEST.sha256 --quiet ); then
    echo "All checksums match. The archive is complete and undamaged."
else
    echo >&2
    echo "Checksum verification FAILED. Do not use this copy: re-download or re-unzip." >&2
    exit 1
fi
