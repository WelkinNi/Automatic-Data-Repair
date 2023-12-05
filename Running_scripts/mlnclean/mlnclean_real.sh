#!/bin/bash

# Define variables
TASK_LIST=("flights" "rayyan" "hospital" "beers")
# TASK="hospital"
CNTS_LIST=(1 2 3)
NUMS_LIST=('01')
for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            DIRTY_DATA="${TASK}/dataset/"
            TASK_NAME="${TASK}-inner_outer_error-${NUM}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${TASK_NAME}"
            echo "Hello, world!"
            echo "${DIRTY_DATA_PATH}"
            /usr/bin/env /usr/lib/jvm/java-8-openjdk-amd64/bin/java -cp /tmp/cp_21mkjr659r871u00zg8emw14c.jar main.Test "${DIRTY_DATA_PATH}" "trainData.csv" "testData.csv" 1 0 || true
        done
    done
done