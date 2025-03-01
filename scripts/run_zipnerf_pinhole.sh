# TODO: need to convert pinhole to equi to use the code

python train.py \
    -m output/zipnerf/undistorted/berlin \
    -s /mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/berlin \
    --images images_4 \
    --iterations 30000 \
    --save_iterations 10000 20000 30000\
    --test_iterations 10000 20000 30000 \
    --bs 3 \
    -r 1 \
    --sh_degree 3 \
    --camera_model PINHOLE \
    --train_random_background \

python render.py \
    -m output/zipnerf/undistorted/berlin \
    -s /mnt/data_ssd_4tb/Datasets/zipnerf/undistorted/berlin \
    --iteration 30000 \
    --camera_model PINHOLE \
    -r 1  \
    --skip_train \

# TODO: convert equi back to pinhole

python metrics.py \
    -m output/zipnerf/undistorted/berlin \