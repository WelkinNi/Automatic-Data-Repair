#!/bin/bash

# Define variables
TASK_LIST=("hospital" "flights" "beers" "rayyan")
# TASK="hospital"
NUMS_LIST=('01')
CNTS_LIST=(1 2 3)

# Fixing for hospital
CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data with dc_rules/hospital/clean.csv"
RULE="/data/nw/DC_ED/References/DATASET/data with dc_rules/hospital/dc_rules-validate-fd-horizon.txt"
PYTHON="/data/nw/DC_ED/References_inner_and_outer/Unified/Unified.py"
TASK_NAME="hospital1"
DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data with dc_rules/hospital/noise/hospital-inner_outer_error-"
DIRTY_DATA_PATH="/data/nw/DC_ED/References/DATASET/data with dc_rules/hospital/noise/hospital-inner_outer_error-50.csv"
/home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &

for CNT in "${CNTS_LIST[@]}"
do
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data with dc_rules/${TASK}/clean.csv"
            RULE="/data/nw/DC_ED/References/DATASET/data with dc_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
            
            PYTHON="/data/nw/DC_ED/References_inner_and_outer/Unified/Unified.py"
            TASK_NAME="${TASK}${CNT}"
            
            DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data with dc_rules/${TASK}/noise/${TASK}-inner_outer_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
    done
    # wait
done


