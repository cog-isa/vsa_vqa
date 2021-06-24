# RUN PRETRAIN

python tools/run_train.py \
    -- ppo 1 
    --checkpoint_every 200 \
    --num_iters 5000 \
    --run_dir ../data/reason/outputs/model_pretrain_ppo_lstm \
    --clevr_train_question_path ../data/reason/clevr_h5/clevr_train_3questions_per_family.h5

CUDA_VISIBLE_DEVICES=0 python tools/run_train.py \
    --checkpoint_every 200 \
    --num_iters 5000 \
    --run_dir ../data/reason/outputs/model_pretrain_split_heads_new \
    --clevr_train_question_path ../data/reason/clevr_h5/clevr_train_3questions_per_family.h5

# RUN REINFORCE 

CUDA_VISIBLE_DEVICES=0 python tools/run_train.py \
    --vsa 1 \
    --reinforce 1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --checkpoint_every 10000 \
    --num_iters 1000000 \
    --run_dir ../data/reason/outputs/model_reinforce_vsa_bs2_15k_FINAL \
    --load_checkpoint_path ../data/reason/outputs/model_pretrain_uniform_270pg/checkpoint.pt \
    --visualize_training 1

CUDA_VISIBLE_DEVICES=0 python tools/run_train.py \
    --reinforce 1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --checkpoint_every 10000 \
    --num_iters 2000000 \
    --run_dir ../data/reason/outputs/model_reinforce_split_heads_final \
    --load_checkpoint_path ../data/reason/outputs/model_pretrain_split_heads_new/checkpoint_best.pt \
    --visualize_training 1

# RUN PPO

CUDA_VISIBLE_DEVICES=1 python tools/run_train.py \
    --ppo 1 \
    --learning_rate 1e-5 \
    --checkpoint_every 2000 \
    --num_iters 1000000 \
    --run_dir ../data/reason/outputs/model_reinforce_ppo_check \
    --load_checkpoint_path ../data/reason/outputs/model_pretrain_ppo/checkpoint.pt \
    --visualize_training 1

# RUN TEST 

CUDA_VISIBLE_DEVICES=1 python tools/run_test.py \
    --run_dir ../data/reason/results_split_heads \
    --load_checkpoint_path ../data/reason/outputs/model_reinforce_split_heads_final/checkpoint_best.pt \
    --clevr_val_scene_path ../data/attr_net/results/clevr_val_scenes_zerotrained.json \
    --save_result_path ../data/reason/results_split_heads_final/result.json

# RUN VSA TEST 

CUDA_VISIBLE_DEVICES=2 python tools/run_test_vsa.py \
    --run_dir ../data/reason/results_bs_1 \
    --load_checkpoint_path ../data/reason/outputs/model_reinforce_vsa_bs2_15k_minerl/checkpoint_best.pt  \
    --clevr_val_scene_path ../data/attr_net/results/clevr_val_scenes_zerotrained.json \
    --save_result_path ../data/reason/results_minerl_15k/result.json

# RUN VSA PRETRAIN

CUDA_VISIBLE_DEVICES=1 python tools/run_train.py \
    --vsa_pretrain 1 \
    --reinforce 1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --checkpoint_every 10000 \
    --num_iters 1000000 \
    --vsa_pretrain_steps 200000 \
    --run_dir ../data/reason/outputs/model_reinforce_vsa_bs2_pretrain_200k \
    --load_checkpoint_path ../data/reason/outputs/model_pretrain_uniform_270pg/checkpoint.pt \
    --visualize_training 1
