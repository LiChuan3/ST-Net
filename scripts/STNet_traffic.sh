export CUDA_VISIBLE_DEVICES=0,1,2

model_name=STNet

seq_len=96
e_layers=3
num_layers_intra_trend=1
num_layers_intra_season=1
season_top_k=3
num_kernels=3
num_experts=5
patch_sizes=(12 8 6 4 2)
choose_k=3
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.005
d_model=16
d_ff=32
batch_size=16
devices="0,1,2"
  
python -u ../run.py \
  --is_training 1 \
  --root_path ../dataset/traffic/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --num_layers_intra_trend $num_layers_intra_trend \
  --num_layers_intra_season $num_layers_intra_season \
  --season_top_k $season_top_k \
  --num_kernels $num_kernels \
  --num_experts $num_experts \
  --patch_sizes "${patch_sizes[@]}" \
  --choose_k $choose_k \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window \
  --down_sampling_method avg \
  --use_multi_gpu \
  --devices $devices \
  --lradj 'OC' \