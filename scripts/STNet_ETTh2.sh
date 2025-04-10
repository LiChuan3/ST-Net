export CUDA_VISIBLE_DEVICES=0

model_name=STNet

seq_len=96
e_layers=3
num_layers_intra_trend=1
num_layers_intra_season=1
season_top_k=3
num_kernels=3
num_experts=3
patch_sizes=(8 6 4)
choose_k=1
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=16
batch_size=128
top_k=5

python -u ../run.py \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers $e_layers \
  --num_layers_intra_trend $num_layers_intra_trend \
  --num_layers_intra_season $num_layers_intra_season \
  --season_top_k $season_top_k \
  --num_kernels $num_kernels \
  --num_experts $num_experts \
  --patch_sizes "${patch_sizes[@]}" \
  --choose_k $choose_k \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window \
  --down_sampling_method conv \
  --top_k $top_k \


python -u ../run.py \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers $e_layers \
  --num_layers_intra_trend $num_layers_intra_trend \
  --num_layers_intra_season $num_layers_intra_season \
  --season_top_k $season_top_k \
  --num_kernels $num_kernels \
  --num_experts $num_experts \
  --patch_sizes "${patch_sizes[@]}" \
  --choose_k $choose_k \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window \
  --down_sampling_method conv \
  --top_k $top_k \

python -u ../run.py \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers $e_layers \
  --num_layers_intra_trend $num_layers_intra_trend \
  --num_layers_intra_season $num_layers_intra_season \
  --season_top_k $season_top_k \
  --num_kernels $num_kernels \
  --num_experts $num_experts \
  --patch_sizes "${patch_sizes[@]}" \
  --choose_k $choose_k \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window \
  --down_sampling_method conv \
  --top_k $top_k \

python -u ../run.py \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers $e_layers \
  --num_layers_intra_trend $num_layers_intra_trend \
  --num_layers_intra_season $num_layers_intra_season \
  --season_top_k $season_top_k \
  --num_kernels $num_kernels \
  --num_experts $num_experts \
  --patch_sizes "${patch_sizes[@]}" \
  --choose_k $choose_k \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_window $down_sampling_window \
  --down_sampling_method conv \
  --top_k $top_k \

