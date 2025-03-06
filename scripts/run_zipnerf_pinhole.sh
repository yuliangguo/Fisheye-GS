DATASET_PATH="/mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/alameda"
OUTPUT_PATH="output/zipnerf/undistorted/alameda"

python prepare_zipnerf_pers2equi.py \
    --path $DATASET_PATH \
    --src images_4 \
    --dst images_4_equidist

python train.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --images images_4_equidist \
    --iterations 30000 \
    --save_iterations 10000 20000 30000\
    --test_iterations 10000 20000 30000 \
    --bs 3 \
    -r 1 \
    --sh_degree 3 \
    --camera_model FISHEYE \
    --train_random_background \

# render
python render.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --iteration 30000 \
    --camera_model FISHEYE \
    -r 1  \
    --skip_train \

# wrap back to origianal space
python prepare_zipnerf_equi2pers.py \
    --camera-path $DATASET_PATH/sparse/0/cameras.bin \
    --src $OUTPUT_PATH/test/ours_30000/gt \
    --dst $OUTPUT_PATH/test/ours_30000/gt_remap \
    -r 4

python prepare_zipnerf_equi2pers.py \
    --camera-path $DATASET_PATH/sparse/0/cameras.bin \
    --src $OUTPUT_PATH/test/ours_30000/renders \
    --dst $OUTPUT_PATH/test/ours_30000/renders_remap \
    -r 4

# evaluation
python metrics.py \
    -m $OUTPUT_PATH \
    --use_remap

# render cross camera
python render.py \
    -m $OUTPUT_PATH \
    -s $DATASET_PATH \
    --iteration 30000 \
    --camera_model FISHEYE \
    -r 1 \
    --skip_train \
    --cross_camera \
    --images images_8_equidist

# wrap back to origianal space
python prepare_zipnerf_equi2fish.py \
    --camera-path $DATASET_PATH/sparse/0/cameras.bin \
    --src $OUTPUT_PATH/test/ours_30000/gt_cross_camera \
    --dst $OUTPUT_PATH/test/ours_30000/gt_cross_camera_remap \
    -r 8 \
    --cross_camera

python prepare_zipnerf_equi2fish.py \
    --camera-path $DATASET_PATH/sparse/0/cameras.bin \
    --src $OUTPUT_PATH/test/ours_30000/renders_cross_camera \
    --dst $OUTPUT_PATH/test/ours_30000/renders_cross_camera_remap \
    -r 8 \
    --cross_camera

# evaluation
python metrics.py \
    -m $OUTPUT_PATH \
    --use_remap \
    --cross_camera
