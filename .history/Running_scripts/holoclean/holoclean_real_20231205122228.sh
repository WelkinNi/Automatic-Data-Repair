#!/bin/bash
TASK_LIST=("hospital" "flights" "beers" "rayyan")
CNTS_LIST=(1 2 3)
for CNT in "${CNTS_LIST[@]}"
do
    NUMS_LIST=('01')
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/holoclean-master/holoclean_run.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for TASK in "${TASK_LIST[@]}"
    do        
        for NUM in "${NUMS_LIST[@]}"
        do                
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA="./data with dc_rules/${TASK}/noise/${TASK}-inner_outer_error-"
            CLEAN_DATA="./data with dc_rules/${TASK}/clean.csv"
            RULE="./data with dc_rules/${TASK}/dc_rules_holoclean.txt"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA"
            wait
        done
    done
done

# bash ./Running_scripts_inner_and_outer/holoclean/running_holoclean_perfected.sh
