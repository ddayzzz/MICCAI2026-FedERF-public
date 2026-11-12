
if [[ $# -ne 6 ]];
then

echo "<algo> <model> <gpu> <start_fold_id> <run_times> <eval_site>"
exit 1

fi
export PYTHONPATH=$PYTHONPATH:${PWD}

all_client_names='LIHC SARC SKCM STAD UCEC'
lr=0.0002
wd=0.00001
bs=1  # 默认16
base_seed=2025
global_round=30
local_epoch=1
eval_gap=10  #
prev=$4
gpu=$3
run_times=$5
eval_site=$6
algo=$1
data='tcga'
model=$2 # 'ab_mil'
num_clients=4
joint_ratio=1.0
optimizer=adamw
lr_decay='cosine'
lr_warmup_round="1"
lr_end=0.0
#train_args="-data ${data} -m ${model} -ecn ${eval_site} --test_global_model_names ${eval_site} --num_classes 4 -t ${run_times} -pv ${prev} --seed ${base_seed} -nc ${num_clients} -jr ${joint_ratio} -opn ${optimizer}  --eval_gap ${eval_gap} -gr ${global_round} -ls ${local_epoch} -did ${gpu} -lbs ${bs} -lr ${lr} -wd ${wd} -acn ${all_client_names}  -ldw ${lr_warmup_round} -ld ${lr_decay} "
# 不要decay
train_args="-data ${data} -m ${model} -ecn ${eval_site} --test_global_model_names ${eval_site} --num_classes 4 -t ${run_times} -pv ${prev} --seed ${base_seed} -nc ${num_clients} -jr ${joint_ratio} -opn ${optimizer}  --eval_gap ${eval_gap} -gr ${global_round} -ls ${local_epoch} -did ${gpu} -lbs ${bs} -lr ${lr} -wd ${wd} -lr_end ${lr_end} -acn ${all_client_names}"

exp_name=glb${global_round}_loc${local_epoch}_lr${lr}_wd${wd}_opt${optimizer}_nc${num_clients}jr${joint_ratio}_bs${bs}_noLRdecay
log_dir="${PWD}/logs/data_${data}_model_${model}/eval_site_${eval_site}"
mkdir -p ${log_dir}

if [[ ${algo} = "local" ]];then

python main.py -algo Local ${train_args} --goal ${exp_name} > ${log_dir}/result-${algo}_${exp_name}.log 2>&1


fi

if [[ ${algo} = "fedavg" ]];then

python main.py -algo FedAvg ${train_args} --goal ${exp_name} > ${log_dir}/result-${algo}_${exp_name}.log 2>&1

fi

if [[ ${algo} = "fedhisto" ]];then

python main.py -algo FedHisto ${train_args} --goal ${exp_name} > ${log_dir}/result-${algo}_${exp_name}.log 2>&1

fi


if [[ ${algo} = "moon" ]];then

tau=1
mu=1

python main.py -algo MOON ${train_args} --goal ${exp_name}_tau${tau}_mu${mu} -tau ${tau} -mu ${mu}  > ${log_dir}/result-${algo}_${exp_name}_tau${tau}_mu${mu}.log 2>&1

fi

if [[ ${algo} = "fedprox" ]];then

mu=0.005

python main.py -algo FedProx ${train_args} --goal ${exp_name}_mu${mu} -mu ${mu}  > ${log_dir}/result-${algo}_${exp_name}_mu${mu}.log 2>&1

fi

if [[ ${algo} = "roupkt" ]];then

checkpoint="models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold{fold_id}.pt"
python main.py -algo ROUPKT ${train_args} --goal ${exp_name} --checkpoint ${checkpoint}  > ${log_dir}/result-${algo}_${exp_name}.log 2>&1

fi

if [[ ${algo} = "roupkt2" ]];then

checkpoint="/homeb/wangshu/projects2/fedmed2026/models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold{fold_id}.pt"
python main.py -algo ROUPKT2 ${train_args} --goal ${exp_name} --checkpoint ${checkpoint}  > ${log_dir}/result-${algo}_${exp_name}.log 2>&1

fi


if [[ ${algo} = "our0209" ]];then

new_expname="${exp_name}_0209_loraLast_finetune30Epc"
python main.py -algo Our ${train_args} --goal ${new_expname} > ${log_dir}/result-${algo}_${new_expname}.log 2>&1


fi



