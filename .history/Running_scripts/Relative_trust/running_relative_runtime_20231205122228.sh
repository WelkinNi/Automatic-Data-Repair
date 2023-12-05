#!/bin/bash
TASK_LIST=("tax")
CNTS_LIST=(1)
NUMS_LIST=($(seq -w 1 1 5 | xargs printf "%04dk\n"))

for CNT in "${CNTS_LIST[@]}"
do
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/Relative_Trust/relative.py"
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do
        echo ${NUM}
        for TASK in "${TASK_LIST[@]}"
        do 
            DIRTY_DATA="./data with dc_rules/tax/split_data/tax-dirty-original_error-${NUM}.csv"
            CLEAN_DATA="./data with dc_rules/tax/split_data/tax-clean-clean_data_ori-${NUM}.csv"
            RULE="./data with dc_rules/${TASK}/dc_rules-validate-fd-horizon.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH=${DIRTY_DATA}
            
            COMMAND="timeout 1d /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name \"$TASK_NAME\" --rule_path \"$RULE\" --onlyed 0 --perfected 0 --dirty_path \"$DIRTY_DATA_PATH\" --clean_path \"$CLEAN_DATA\" \&"
            { eval "$COMMAND" || echo "Timeout reached: $(date) ${COMMAND}" ; } 2>&1 | tee -a "./aggre_results/timeout_log.txt" 
        done
    done
done


