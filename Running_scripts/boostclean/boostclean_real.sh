#!/bin/bash
TASK_LIST=("hospital" "flights" "rayyan" "beers")
CNTS_LIST=(1 2 3)
NUMS_LIST=('01')

for CNT in "${CNTS_LIST[@]}"
do
    PYTHON="./BoostClean/activedetect/experiments/Experiment.py"
    for NUM in "${NUMS_LIST[@]}"
    do
        echo ${NUM}
        for TASK in "${TASK_LIST[@]}"
        do 
            DIRTY_DATA="./data_with_rules/${TASK}/noise/${TASK}-outer_error-${NUM}.csv"
            CLEAN_DATA="./data_with_rules/${TASK}/clean.csv"
            RULE="./data_with_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH=${DIRTY_DATA}
            /home/dell/anaconda3/envs/tf115/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
            
        done
        wait
    done
done





