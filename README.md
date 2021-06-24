# ВАЖНО !!! 
не забудь починить пути к файлам (где sys.path.append) \
управляй размером VSA векторов при помощи переменной HD_DIM (а лучше допиши в конфиг, чтобы задавать командой)

для визуализации добавь --visualise 1 (пишет логи в тензорборд)

# Примеры скриптов для запусков (правь пути под себя):

## PREPROCESS
```
python3 tools/preprocess_questions.py \
     --input_questions_json ../data/raw/visdial/raw/visdial_unannotated_questions_trainval.json \
     --output_h5_file ../data/raw/visdial/h5/trash.h5 \
     --output_vocab_json ../data/raw/visdial/raw/visdial_vocab.json 
     
python3 tools/preprocess_questions.py \
     --input_questions_json ../data/raw/visdial/raw/visdial_annotated_questions_trainval.json \
     --output_h5_file ../data/raw/visdial/h5/visdial_annotated_questions_trainval.h5 \
     --input_vocab_json ../data/raw/visdial/raw/visdial_vocab.json
     
python3 tools/preprocess_questions.py \
     --input_questions_json ../data/raw/visdial/raw/visdial_unannotated_questions_train.json \
     --output_h5_file ../data/raw/visdial/h5/visdial_unannotated_questions_train.h5 \
     --input_vocab_json ../data/raw/visdial/raw/visdial_vocab.json

python3 tools/preprocess_questions.py \
     --input_questions_json ../data/raw/visdial/raw/visdial_unannotated_questions_val.json \
     --output_h5_file ../data/raw/visdial/h5/visdial_unannotated_questions_val.h5 \
     --input_vocab_json ../data/raw/visdial/raw/visdial_vocab.json
```

## PRETRAIN
```
python3 tools/run_train.py \
    --checkpoint_every 200 \
    --num_iters 5000 \
    --batch_size 4 \
    --run_dir ../data/reason/outputs/model_visdial_bigger_pretrain \
    --clevr_train_question_path ../data/raw/visdial/h5/visdial_annotated_questions_trainval.h5 \
    --clevr_val_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_val.h5 \
    --clevr_vocab_path ../data/raw/visdial/raw/visdial_vocab.json \
    --clevr_train_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --clevr_val_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json
```

## FINETUNE
```
python3 tools/run_train.py \
    --visualise 1 \
    --vsa 1 \
    --reinforce 1 \
    --learning_rate 1e-5 \
    --checkpoint_every 2000 \
    --num_iters 1000000 \
    --run_dir ../data/reason/outputs/model_visdial_rl_bigger_pretrain_vsa \
    --load_checkpoint_path ../data/reason/outputs/model_visdial_bigger_pretrain/checkpoint_best.pt \
    --clevr_train_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_train.h5 \
    --clevr_val_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_val.h5 \
    --clevr_vocab_path ../data/raw/visdial/raw/visdial_vocab.json \
    --clevr_train_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --clevr_val_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json 
```

## TEST
```
python3 tools/run_test.py \
    --run_dir ../data/reason/results \
    --load_checkpoint_path ../data/reason/outputs/model_visdial_rl_bigger_pretrain/checkpoint_best.pt  \
    --clevr_vocab_path ../data/raw/visdial/raw/visdial_vocab.json \
    --clevr_train_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_train.h5 \
    --clevr_val_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_val.h5 \
    --clevr_train_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --clevr_val_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --save_result_path ../data/reason/results/result_vidial.json \
    --batch_size 2 
```

## VSA TEST 
```
python3 tools/run_test_vsa.py \
    --run_dir ../data/reason/results \
    --load_checkpoint_path ../data/reason/outputs/model_visdial_rl_bigger_pretrain_vsa/checkpoint_best.pt  \
    --clevr_vocab_path ../data/raw/visdial/raw/visdial_vocab.json \
    --clevr_train_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_train.h5 \
    --clevr_val_question_path ../data/raw/visdial/h5/visdial_unannotated_questions_val.h5 \
    --clevr_train_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --clevr_val_scene_path ../data/raw/visdial/raw/coco_trainval_scenes.json \
    --save_result_path ../data/reason/results/result_vidial.json \
    --batch_size 2 
``` 
