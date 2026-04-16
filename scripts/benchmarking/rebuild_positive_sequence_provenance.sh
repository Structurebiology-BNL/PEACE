#!/usr/bin/env bash

set -euo pipefail

# Rebuild positive-sequence provenance intermediates without host-specific paths.
# Expected inputs live under src/data/dataset_construction/positive_seqs.

WORK_DIR="${1:-src/data/dataset_construction}"
POSITIVE_DIR="${2:-${WORK_DIR}/positive_seqs}"
THREADS="${THREADS:-16}"
CD_HIT_BIN="${CD_HIT_BIN:-cd-hit}"
PSI_CD_HIT_BIN="${PSI_CD_HIT_BIN:-psi-cd-hit.pl}"
CLSTR_REV_BIN="${CLSTR_REV_BIN:-clstr_rev.pl}"

cd "${POSITIVE_DIR}"

cat *.fasta > ../combined_positives.fasta
seqkit rmdup -s -i -o ../combined_positives_deduplicated.fasta ../combined_positives.fasta

cd "${WORK_DIR}"

"${CD_HIT_BIN}" -i combined_positives_deduplicated.fasta -o combined_90 -c 0.9 -n 5 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T "${THREADS}" -M 0 > combined_90.log
"${CD_HIT_BIN}" -i combined_90 -o combined_75 -c 0.75 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T "${THREADS}" -M 0 > combined_75.log
"${CD_HIT_BIN}" -i combined_75 -o combined_60 -c 0.6 -n 4 -g 1 -G 0 -aS 0.8 -d 0 -p 1 -T "${THREADS}" -M 0 > combined_60.log

"${PSI_CD_HIT_BIN}" -i combined_60 -o combined_50 -c 0.50 -ce 1e-6 -aS 0.8 -G 0 -g 1 -exec local -para 8 -blp 4
"${PSI_CD_HIT_BIN}" -i combined_50 -o combined_40 -c 0.40 -ce 1e-6 -aS 0.8 -G 0 -g 1 -exec local -para 8 -blp 4

"${CLSTR_REV_BIN}" combined_90.clstr combined_75.clstr > combined90-75.clstr
"${CLSTR_REV_BIN}" combined90-75.clstr combined_60.clstr > combined90-75-60.clstr
"${CLSTR_REV_BIN}" combined90-75-60.clstr combined_50.clstr > combined90-75-60-50.clstr
"${CLSTR_REV_BIN}" combined90-75-60-50.clstr combined_40.clstr > combined90-75-60-50-40.clstr

