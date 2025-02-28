export CUDA_VISIBLE_DEVICES=0

python render.py \
    -m output/0a5c013435 \
    -s /mnt/data_ssd_4tb/Datasets/scannetpp_tiny/data/0a5c013435/dslr \
    --iteration 30000 \
    --camera_model FISHEYE \
    -r 1  