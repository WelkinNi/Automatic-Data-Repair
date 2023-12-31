#!/bin/bash
TASK_LIST=("tax")
CNTS_LIST=(1 2 3)
NUMS_LIST=()

for ((i=10; i<=10; i+=10)); 
do
  NUMS_LIST+=("$(printf "%04d" $i)k") 
done

for CNT in "${CNTS_LIST[@]}"
do
    PYTHON="./BigDansing_Holistic/holistic.py"
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do
        echo ${NUM}
        for TASK in "${TASK_LIST[@]}"
        do 
            DIRTY_DATA="./data_with_rules/tax/split_data/tax-dirty-original_error-${NUM}.csv"
            CLEAN_DATA="./data_with_rules/tax/split_data/tax-clean-clean_data_ori-${NUM}.csv"
            RULE="./data_with_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH=${DIRTY_DATA}
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --rule_path "$RULE" --onlyed 0 --perfected 0 --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
    done
    wait
done


