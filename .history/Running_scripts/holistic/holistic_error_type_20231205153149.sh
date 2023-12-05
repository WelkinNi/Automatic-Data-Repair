#!/bin/bash
TASK_LIST=("hospital" "beers" "rayyan" "flights")
CNTS_LIST=(2 3)

PYTHON="/data/nw/DC_ED/References_inner_and_outer/BigDansing/holistic.py"
DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/noise/hospital-outer_error-"
CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/clean.csv"
RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/dc_rules_holoclean.txt"
TASK_NAME="hospital2"
DIRTY_DATA_PATH="${DIRTY_DATA}70.csv"
/home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &

PYTHON="/data/nw/DC_ED/References_inner_and_outer/BigDansing/holistic.py"
DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/noise/hospital-outer_error-"
CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/clean.csv"
RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/hospital/dc_rules_holoclean.txt"
TASK_NAME="hospital3"
DIRTY_DATA_PATH="${DIRTY_DATA}70.csv"
/home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &


for CNT in "${CNTS_LIST[@]}"
do
    NUMS_LIST=(90)
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
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &

            DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-outer_error-"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
    done
    wait
done






