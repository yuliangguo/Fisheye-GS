export CUDA_VISIBLE_DEVICES=0
SCENE_ID=1d003b07bd #2a1a3afad9 #2a1a3afad9 #1f7cbbdde1 4ef75031e3 1d003b07bd
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/
# OUTPUT_DIR=./output_scannetpp_fs_gt/dslr/$SCENE_ID
OUTPUT_DIR=./output_fullfov_updated/scannetpp/$SCENE_ID


# python prepare_scannetpp_fish2equi.py \
#     --path $DATASET_DIR \
#     --src resized_images \
#     --dst images_equidist

python render.py \
    -m $OUTPUT_DIR \
    -s $DATASET_DIR \
    --iteration 30000 \
    --camera_model FISHEYE \
    -r 1 \
    --skip_train