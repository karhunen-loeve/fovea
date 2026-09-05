#!/bin/sh
#
# Published sources must not cite workspace-internal documents.
#
# PHILOSOPHY.md, the ADRs and the Roadmap live in the private workspace
# repository. They are not part of the published crate, so a reference to
# them in rustdoc, benches or examples dangles for every reader outside the
# workspace: docs.rs serves the rustdoc, and `benches/**/*.rs` is in the
# crate's `include` list. Inline the principle as the reasoning it stands
# for instead of naming the document.
#
# The rule was announced in the 0.3.0 changelog and had regressed twice by
# the end of the 0.4.0 cycle (v0.4.0 review, finding A8), which is what a
# gate is for.
#
# Usage: check-no-internal-citations.sh
#
# Run from the repository root. Exits 0 when no published source cites an
# internal document, 1 otherwise.

set -eu

status=0
for pattern in 'PHILOSOPHY' 'ADR-[0-9][0-9][0-9][0-9]' '[Rr]oadmap'; do
    if matches=$(grep -rn -E "$pattern" src benches examples 2>/dev/null); then
        echo "FAIL  published sources match '$pattern':" >&2
        printf '%s\n' "$matches" >&2
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "No internal-document citations in published sources."
else
    echo >&2
    echo "PHILOSOPHY.md, the ADRs and the Roadmap are not part of the" >&2
    echo "published crate. Inline the reasoning the citation stands for" >&2
    echo "instead of naming the document." >&2
fi

exit "$status"
