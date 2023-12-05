#!/bin/bash

# Define variables
TASK_LIST=("hospital" "beers" "rayyan" "flights")
# TASK="hospital"
for TASK in "${TASK_LIST[@]}"
do
    DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-inner_outer_error-"
    CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/clean.csv"
    RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
    NUMS_LIST=(10 30 50 70 90)
    CNTS_LIST=(1 2 3)
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/Relative_Trust/relative.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for CNT in "${CNTS_LIST[@]}"
        do                
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            
            COMMAND="timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name \"$TASK_NAME\" --rule_path \"$RULE\" --onlyed 0 --perfected 0 --dirty_path \"$DIRTY_DATA_PATH\" --clean_path \"$CLEAN_DATA\" \&"
            { eval "$COMMAND" || echo "Timeout reached: $(date) ${COMMAND}" ; } 2>&1 | tee -a "./aggre_results/timeout_log.txt" 
        done
    done
done

