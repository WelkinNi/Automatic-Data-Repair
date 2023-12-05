#!/bin/bash

# Define variables
TASK_LIST=("tax")
# TASK="hospital"
CNTS_LIST=(1)
NUMS_LIST=('10k' '20k' '30k' '40k' '50k')
for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            DIRTY_DATA="${TASK}/dataset/"
            TASK_NAME="${TASK}-dirty-original_error-00${NUM}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${TASK_NAME}"
            echo "Hello, world!"
            echo "${DIRTY_DATA_PATH}"
            cd /data/nw/DC_ED/References_inner_and_outer/mlnclean/code/MLNClean; 
            time /usr/bin/env /usr/lib/jvm/java-8-openjdk-amd64/bin/java -cp /tmp/cp_21mkjr659r871u00zg8emw14c.jar main.Test "${DIRTY_DATA_PATH}" "trainData.csv" "testData.csv" 1 0 > ./aggre_results/output.txt 2>&1 || true
        done
    done
done