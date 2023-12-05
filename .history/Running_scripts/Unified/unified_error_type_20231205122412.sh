#!/bin/bash

# Define variables
TASK_LIST=("beers" "flights")
# TASK="hospital"
NUMS_LIST=(50 70 90)
CNTS_LIST=(1)

# Fixing for flights
CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/clean.csv"
RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/dc_rules-validate-fd-horizon.txt"
PYTHON="/data/nw/DC_ED/References_inner_and_outer/Unified/Unified.py"
TASK_NAME="flights1"
DIRTY_DATA_PATH="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/noise/hospital-outer_error-30.csv"
timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &

for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/clean.csv"
            RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
            
            PYTHON="/data/nw/DC_ED/References_inner_and_outer/Unified/Unified.py"
            TASK_NAME="${TASK}${CNT}"

            DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-outer_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
    done
    wait
done


