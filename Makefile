clean:
	docker rm -f $$(docker ps -qa)
build:
	docker build --build-arg IMAGE_NAME=nvidia/cuda -t sampledocker .

cleandangling:
  docker rmi -f $(docker images -f "dangling=true" -q)

run:
	docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/home/hope/docker_debug:/app" \
        --volume="/mnt/data/rec_data:/data" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="sampledocker" \
      	sampledocker bash
