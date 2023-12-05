#!/bin/bash
TASK_LIST=("hospital" "beers" "rayyan" "flights")
CNTS_LIST=(1 2 3)
for CNT in "${CNTS_LIST[@]}"
do
    NUMS_LIST=(10 30 50 70 90)
    PYTHON="./holoclean-master/holoclean_run.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do                
            TASK_NAME="${TASK}${CNT}"
            CLEAN_DATA="./data_with_rules/${TASK}/clean.csv"
            RULE="./data_with_rules/${TASK}/dc_rules_holoclean.txt"

            DIRTY_DATA="./data_with_rules/${TASK}/noise/${TASK}-inner_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" 
            wait

            DIRTY_DATA="./data_with_rules/${TASK}/noise/${TASK}-outer_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" 
            wait
        done
        wait
    done
done
