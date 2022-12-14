task="answer-generation"
device="0,1"

###### Model Options ######
#model="facebook/bart-base"
#model="facebook/bart-large"
#model="t5-base"
#model="allenai/unifiedqa-t5-base"
model="allenai/unifiedqa-t5-large"


###### Additional Model Suffix ######
suffix=""

lrs=(5e-5 1e-4 2e-4)
batch=(2)
seeds=(5 7 23)
for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
        output_dir="./output/${model}_batch_${s}_lr_${l}_seed_${seed}${suffix}_prefix_transE/"
        # output_dir="./output/${model}_prefix_transE/"
        python ./code/eval_ans_gen.py \
        --data_dir "./data/" \
        --model ${model} \
        --task_name  ${task} \
        --file_suffix "_ans_gen.json" \
        --device_num ${device} \
        --eval_batch_size 8 \
        --num_train_epochs 10 \
        --max_seq_length 340 \
        --learning_rate ${l} \
        --seed ${seed} \
        --model_dir ${output_dir}
        done
    done
done
