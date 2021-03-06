#!/bin/bash

NEG_REWARD=(1 0.5 0.25 0.125 0.01 0 -0.05 -0.125 -0.5 -1 -2 -4 -8)
RHO=(-2 -1 -0.5 -0.25 -0.125 -0.05)
RHO_2=(-0.01 0 0.01 0.05 0.125 0.25 0.5)

dataset='fb15k'
hidden_unit_list="hidden_unit_list: [90,40]"
logs='grid_search_semi/grid_search_'${dataset}'/90_40'
mkdir -p $logs
training_data_path='../logs/'${dataset}'/sm_with_id.data.pkl'
labelled_training_data_path='../logs/'${dataset}'/exp_words/sm_sup_train_with_id.pkl'
val_data_path='../logs/'${dataset}'/exp_words/sm_sup_valid_with_id.pkl'
base_model_file='../dumps/'${dataset}'_distmult_dump_norm.pkl'
supervision='semi'
for i in "${NEG_REWARD[@]}"
do
	for k in 1 2
	do
		if [ $k == 1 ]
		then
			temp_rho=("${RHO[@]}")
		else
			temp_rho=("${RHO_2[@]}")
		fi
		sh="exp_"$i".sh"
		echo $sh
		cp "single_run.sh" "$sh"
		counter=0
		for j in "${temp_rho[@]}"
		do
			temp="exp_""$i""_""$j"
			yml=$logs/$temp'.yml'
			echo $yml
			cp "config_template.yml" "$yml"
			echo "neg_reward: $i" >> $yml
			echo "rho: $j" >> $yml
			echo "${hidden_unit_list}" >> $yml
			echo "/home/cse/btech/cs1150210/anaconda3/bin/python3 main.py --training_data_path $training_data_path --labelled_training_data_path $labelled_training_data_path --exp_name $temp --output_path $logs --num_epochs 20 --config $yml --lr 0.001 --log_after 500000 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --val_data_path $val_data_path --base_model_file $base_model_file --supervision $supervision &" >> $sh
			echo "pids[${counter}]=""$""!" >> $sh
			counter=$((counter+1))
		done
		echo "for pid in \${pids[*]}; do" >> $sh
		echo "	wait \$pid" >> $sh
		echo "done" >> $sh
		qsub -P cse $sh
		rm $sh
	done
done
