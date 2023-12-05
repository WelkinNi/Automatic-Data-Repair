
#!/bin/bash
TASK_LIST=("tax")
CNTS_LIST=(1 2 3)
for ((i=10; i<=10; i+=2)); 
do
  NUMS_LIST+=("$(printf "%04d" $i)k") 
done

for TASK in "${TASK_LIST[@]}"
do
    PYTHON="/data/nw/DC_ED/References_inner_and_outer/raha-master/correction_with_raha.py"
    # Loop through error rates and run holoclean_run.py on dataset
    for NUM in "${NUMS_LIST[@]}"
    do
        for CNT in "${CNTS_LIST[@]}"
        do 
            echo ${NUM}
            DIRTY_DATA="./data with dc_rules/tax/split_data/tax-dirty-original_error-${NUM}.csv"
            CLEAN_DATA="./data with dc_rules/tax/split_data/tax-clean-clean_data_ori-${NUM}.csv"
            RULE="./data with dc_rules/${TASK}/dc_rules_holoclean.txt"
                    
            TASK_NAME="${TASK}${CNT}"
            DIRTY_DATA_PATH=${DIRTY_DATA}
            
            /home/dell/anaconda3/envs/torch110/bin/python $PYTHON --task_name "$TASK_NAME" --dirty_path "$DIRTY_DATA_PATH" --clean_path "$CLEAN_DATA" &
        done
        wait
    done
done





