#!/bin/bash
TASK_LIST=("hospital" "beers" "rayyan" "flights")
for TASK in "${TASK_LIST[@]}"
do
    DIRTY_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/noise/${TASK}-inner_outer_error-"
    CLEAN_DATA="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/clean.csv"
    RULE="/data/nw/DC_ED/References/DATASET/data_with_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
    NUMS_LIST=('01')
    CNTS_LIST=(1 2 3)
    PYTHON="./BoostClean/activedetect/experiments/Experiment.py"

    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do        
        for CNT in "${CNTS_LIST[@]}"
        do                
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH="${DIRTY_DATA}${NUM}.csv"
            timeout 1d /home/dell/anaconda3/envs/tf115/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 1 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" || true     
        done
        # wait
    done
done


