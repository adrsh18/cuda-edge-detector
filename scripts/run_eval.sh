#!/bin/bash

BASE_DIR=$(dirname $0)
FILES=${BASE_DIR}/../eval_images/*
OUT_DIRECTORY=${BASE_DIR}/../eval_out
PARALLEL=${BASE_DIR}/../edge_detector
SERIAL=${BASE_DIR}/../edge_detector_serial
HIGH_THRESHOLD=100
LOW_THRESHOLD=50

if [ -d "${OUT_DIRECTORY}" ]; then
    rm -fr ${OUT_DIRECTORY}
fi

mkdir ${OUT_DIRECTORY}

for f in $FILES
do
    OUT_FILE=$(basename "$f")
    echo ""
    echo "Processing ${OUT_FILE} in Parallel.."
    echo "------------------------------------"
    ${PARALLEL} $f "${OUT_DIRECTORY}/p_out_${OUT_FILE}" ${HIGH_THRESHOLD} ${LOW_THRESHOLD}
    echo "------------------------------------"
    echo ""
    echo "Processing ${OUT_FILE} in Serial.."
    ${SERIAL} $f "${OUT_DIRECTORY}/s_out_${OUT_FILE}" ${HIGH_THRESHOLD} ${LOW_THRESHOLD}
    echo "------------------------------------"
    echo ""
done
