DATASET_PATH="/media/scannetpp/demo/0a7cc12c0e/dslr/"
OUTPUT_PATH="output/scannetpp/dslr/0a7cc12c0e"

# python prepare_scannetpp_fish2equi.py \
#     --path $DATASET_PATH \
#     --src resized_images \
#     --dst images_equidist

# python train.py \
#     -m $OUTPUT_PATH \
#     -s $DATASET_PATH \
#     --images images_equidist \
#     --iterations 30000 \
#     --save_iterations 10000 20000 30000 \
#     --test_iterations 10000 20000 30000 \
#     --bs 3 \
#     -r 1 \
#     --sh_degree 3 \
#     --camera_model FISHEYE \
#     --train_random_background \

# # render
# python render.py \
#     -m $OUTPUT_PATH \
#     -s $DATASET_PATH \
#     --iteration 30000 \
#     --camera_model FISHEYE \
#     -r 1 \
#     --skip_train

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
