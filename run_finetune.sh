
if [[ $# -ne 5 ]];
then

echo "<algo> <model> <gpu> <fold_ids> <eval_site>"
exit 1

fi
export PYTHONPATH=$PYTHONPATH:${PWD}

# ab_mil
lr=0.0001
feature_lr=0.00001
wd=0.00001
end_lr=0
# wikg_mil 调参
#lr=0.0001
#feature_lr=0.0001
#wd=0.00001

epochs=20
fold_ids=$4
gpu=$3
eval_site=$5
algo=$1
model=$2
optimizer=adamw
lr_decay="cosine"
base_args="--dataset ${eval_site} --model ${model} --max_epochs ${epochs} --gpu ${gpu} --feature_lr ${feature_lr} --lr ${lr} --weight_decay ${wd} --lr_decay ${lr_decay} --end_lr ${end_lr}"
for fold_id in ${fold_ids}
do

log_dir=finetune_results/finetune_${eval_site}_${model}/exp_${algo}
mkdir -p ${log_dir}
if [[ ${algo} = "no_pretrain" ]]; then
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fl_pretrain_loadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb20_loc1_lr0.0001_wd0.00001_optadamw_nc7jr1.0_bs1_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} --load_pretrain_classifier > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fl_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb20_loc1_lr0.0001_wd0.00001_optadamw_nc7jr1.0_bs1_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

#if [[ ${algo} = "fl_pretrain_noLoadClf_loc2_lrbackbonelr1e-4" ]]; then
#  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb20_loc2_lr0.0001_wd0.00001_optadamw_nc5jr1.0_bs1_server_fold${fold_id}.pt
#  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1
#
#fi

if [[ ${algo} = "fl_pretrain_noLoadClf_fixbackbone_loc2" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb20_loc2_lr0.0001_wd0.00001_optadamw_nc5jr1.0_bs1_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp}  > ${log_dir}/output.log 2>&1

fi

## 以下是已经确定的版本
if [[ ${algo} = "fedavg_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "feddyn_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedDyn_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_alpha0.005_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "moon_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/MOON_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_tau1_mu1_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fedprox_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedProx_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_mu0.005_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fedhisto_pretrain_noLoadClf" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedHisto_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "roupkt_fedavg_pretrain_noLoadClf" ]]; then

# 借用lora_lr
fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold${fold_id}.pt
new_log_dir=finetune_results/finetune_${eval_site}_${model}_roupkt/exp_${algo}
mkdir -p ${new_log_dir}
python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --model ${model}_roupkt --checkpoint ${fold_fl_pretrain_fp} --lora_lr 1e-4 --lora_wd 1e-5 > ${new_log_dir}/output.log 2>&1

fi

if [[ ${algo} = "roupkt2_fedavg_pretrain_noLoadClf" ]]; then

# 借用lora_lr
fold_fl_pretrain_fp=/homeb/wangshu/projects2/fedmed2026/models/data_tcga_model_${model}/eval_site_${eval_site}/FedAvg_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_server_fold${fold_id}.pt
new_log_dir=finetune_results/finetune_${eval_site}_${model}_roupkt/exp_${algo}
mkdir -p ${new_log_dir}
python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --fold_id ${fold_id} --model ${model}_roupkt --checkpoint ${fold_fl_pretrain_fp} --lora_lr 1e-4 --lora_wd 1e-5 > ${new_log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fl_our0209_pretrain_noLoadClf_loraClf_loraLast_finetune30Epc_softmaxRt_scale_loraLr5e-3" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/Our_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_0209_loraLast_finetune30Epc_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --lora_lr 5e-3 --lora_wd 5e-2 --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi

if [[ ${algo} = "fl_our0209_pretrain_noLoadClf_loraClf_loraLast_finetune30Epc_softmaxRt_scale_loraLr1e-4" ]]; then
  fold_fl_pretrain_fp=models/data_tcga_model_${model}/eval_site_${eval_site}/Our_glb30_loc1_lr0.0002_wd0.00001_optadamw_nc4jr1.0_bs1_noLRdecay_0209_loraLast_finetune30Epc_server_fold${fold_id}.pt
  python flcore/utils/single_mixed.py ${base_args} --exp_name ${algo} --lora_lr 1e-4 --lora_wd 1e-5  --fold_id ${fold_id} --checkpoint ${fold_fl_pretrain_fp} > ${log_dir}/output.log 2>&1

fi
