#!/usr/bin/env bash

EXP_NAME=voc10-5-5
METHOD_NAME=HMWA_2
SAVE_DIR=output/${EXP_NAME}
 # <-- change it to you path
SPLIT_ID=1
IMAGENET_PRETRAIN='./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth'
OFFLINE_RPN_CONFIG='./configs/CED_configs/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml'
BB_RPN_WEIGHTS='./offline_rpn_weights/VOC10-5-5_Percept.pth'
CONCEPT_TXT='./concepts/concepts_voc.txt'

# ------------------------------- Base Pre-train ---------------------------------- #
#python main.py --num-gpus 1 --config-file configs/CED_configs/${EXP_NAME}/CLIP_fast_rcnn_R_50_C4_ovd.yaml --opts OUTPUT_DIR ${SAVE_DIR}/food_r50_voc_base \
#MODEL.WEIGHTS ${IMAGENET_PRETRAIN} MODEL.CLIP.BASE_TRAIN True MODEL.CLIP.OFFLINE_RPN_CONFIG ${OFFLINE_RPN_CONFIG} \
#MODEL.CLIP.BB_RPN_WEIGHTS ${BB_RPN_WEIGHTS} MODEL.CLIP.CONCEPT_TXT ${CONCEPT_TXT}
#


#BASE_WEIGHT='./imgEncoder_baseWeights/VOC10-5-5_Base.pth'
BASE_WEIGHT=${SAVE_DIR}/food_r50_voc_base/'model_final.pth'

#
### ------------------------------ Novel Fine-tuning ------------------------------- #
### --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 1 2 3 4 5 6 7 8 9 10
do
    for shot in 1 3 5 10 # if final, 10 -> 1 2 3 5 10 30
    do
        python tools/create_config.py --dataset voc --config_root configs/CED_configs/voc10-5-5                  \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/CED_configs/voc10-5-5/food_gfsod_r50_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/${METHOD_NAME}/3_3_1/${shot}shot_seed${seed}
        python main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} MODEL.CLIP.CONCEPT_TXT ${CONCEPT_TXT} \
            MODEL.CLIP.BASE_TRAIN False MODEL.CLIP.OFFLINE_RPN_CONFIG ${OFFLINE_RPN_CONFIG} MODEL.CLIP.BB_RPN_WEIGHTS ${BB_RPN_WEIGHTS}  \
            UPLOSS.TOPK 3 UPLOSS.SAMPLING_RATIO 3 UPLOSS.TOPM 1
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done

python tools/extract_results.py --res-dir ${SAVE_DIR}/${METHOD_NAME}/3_3_1 --shot-list 1 3 5 10  # surmarize all results