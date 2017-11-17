#!/bin/bash

BASE_DIR=$(dirname $0)

PLOTS_DIR=${BASE_DIR}/../eval_plots

if [ -d "${PLOTS_DIR}" ]; then
    rm -fr ${PLOTS_DIR}
fi

mkdir ${PLOTS_DIR}

${BASE_DIR}/run_eval.sh | tee ${BASE_DIR}/eval.log
cd ${BASE_DIR}
python plots.py
python plot-split.py

if [[ -f "../eval_plots/serial_vs_parallel.png" && -f "../eval_plots/subroutine_details.png" ]]; then
    echo "Plots generated."
fi

echo "Evaluation complete!"
