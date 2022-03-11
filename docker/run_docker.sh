# Build:
docker build -t daso:v0 .

# Launch (require GPUs)
docker run --gpus all -it \
    --shm-size=12gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name daso daso:v0