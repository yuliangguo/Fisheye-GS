# DATASET_PATH="/mnt/data_ssd_4tb/Datasets/scannetpp_tiny/data/0a5c013435/dslr/"
# OUTPUT_PATH="output/scannetpp/dslr/0a5c013435"
SCENE_ID="steakhouse_patio"
DATASET_PATH="/media/projectaria_tools_aria-scenes_data/scannetpp_formatted/$SCENE_ID/"
OUTPUT_PATH="output/aria/$SCENE_ID"

STEP_EVAL=0.0015
FOVMOD_EVAL=2.0 #0.85 #0.85 #1.0 #2.0 #1.0 #2.0
FOVMAP_DIR_EVAL=undistorted_fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

# python prepare_scannetpp_fish2equi.py \
#     --path $DATASET_PATH \
#     --src resized_images \
#     --dst images_equidist
python prepare_fov.py --path $DATASET_PATH --dst $FOVMAP_DIR_EVAL --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN

python extract_eq.py --path $DATASET_PATH \
                    --src $DATASET_PATH/$FOVMAP_DIR_EVAL \
                    --dst $DATASET_PATH/images_equidist \
                    --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

python train.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --images images_equidist \
    --iterations 30000 \
    --save_iterations 100 10000 20000 30000 \
    --test_iterations 100 10000 20000 30000 \
    --bs 3 \
    -r 1 \
    --sh_degree 3 \
    --camera_model FISHEYE \
    --train_random_background \
    --mask_path $DATASET_PATH/images_equidist/$TEST_MASK_FN \

# render
python render.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --iteration 30000 \
    --camera_model FISHEYE \
    -r 1 \
    --skip_train

# wrap back to origianal space
python prepare_scannetpp_equi2fish.py \
    --camera-path $DATASET_PATH/colmap/cameras_equidist.txt \
    --src $OUTPUT_PATH/test/ours_30000/gt \
    --dst $OUTPUT_PATH/test/ours_30000/gt_remap \

python prepare_scannetpp_equi2fish.py \
    --camera-path $DATASET_PATH/colmap/cameras_equidist.txt \
    --src $OUTPUT_PATH/test/ours_30000/renders \
    --dst $OUTPUT_PATH/test/ours_30000/renders_remap \

# evaluation
python metrics.py \
    -m $OUTPUT_PATH \
    --use_remap
