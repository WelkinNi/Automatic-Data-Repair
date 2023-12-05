#!/bin/bash
TASK_LIST=("hospital" "rayyan")
CNTS_LIST=(1 2 3)

for CNT in "${CNTS_LIST[@]}"
do
    NUMS_LIST=(10 30 50 70 90)
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/BigDansing/holistic.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do
            DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-outer_error-"
            CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/clean.csv"
            RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" 
        done
    done
    wait
done

for CNT in "${CNTS_LIST[@]}"
do
    NUMS_LIST=(10 30 50 70 90)
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/BigDansing/holistic.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for TASK in "${TASK_LIST[@]}"
        do
            DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-inner_error-"
            CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/clean.csv"
            RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA"
        done
    done
    wait
done





