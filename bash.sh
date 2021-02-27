#!/bin/bash

root_dir=/home/yidan/Dropbox/_code_1202

#chmod +x _simulate_data.py

declare -a deg_var_list=(0 1)
declare -a deg_cor_list=(0 1)
declare -a alpha_list=(0.05 0.10 0.25)
declare -a p_list=(0.6 0.7 0.8 0.9 1.0)
declare -a q_list=(0.1 0.2 0.3 0.4 0.5)

declare -a init_type_list=('const' 'sp' 'random-multinomial' 'random-dir' 'z' 'pi' 'perturb-z' 'perturb-pi')

p=${p_list[1]}
q=${q_list[0]}


#p=${p_list[-1]}
#q=${q_list[0]}

#p=${p_list[-1]}
#q=${q_list[1]}

#p=${p_list[-1]}
#q=${q_list[2]}

#p=${p_list[-2]}
#q=${q_list[2]}

#p=${p_list[-3]}
#q=${q_list[2]}

#init_type=${init_type_list[1]}
#init_type=${init_type_list[2]}
#init_type=${init_type_list[3]}
#init_type=${init_type_list[4]}
#init_type=${init_type_list[5]}
#init_type=${init_type_list[6]}
init_type=${init_type_list[7]}


for deg_var in ${deg_var_list[@]}
do
	for deg_cor in ${deg_cor_list[@]}
	do
		for alpha in ${alpha_list[@]}
		do
			save_dir=${root_dir}/result_2021/degvar_${deg_var}_degcor_${deg_cor}_alpha_${alpha}_init_${init_type}_p_${p}_q_${q}
			#save_dir=${root_dir}/result/degvar_${deg_var}_degcor_${deg_cor}_alpha_${alpha}_init_${init_type}_p_${p}_q_${q}
			mkdir -p $save_dir 

			python3 ${root_dir}/_simulate_data.py --deg_var ${deg_var} --deg_cor ${deg_cor} --alpha ${alpha} --p ${p} --q ${q} --init_type ${init_type} --save_dir ${save_dir} &
		done
	done
done

wait 

#exit 0


