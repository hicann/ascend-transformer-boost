# Usage:
# 1. Configure proxy in your host machine.
# (if you want to speed up the docker build procedure, you can put required installation packages under "/data/tmp") 
# 2. bash build_docker.sh
# 3. start_docker.sh [your container name(acltransformer_[your model name]_[your name])]
# 4. docker exec -it --user root [your container name] bash

if [ $# -ne 1 ]; then
	echo "error: need one argument describing your container name."
	echo "usage: $0 [arg], arg in the format of acltransformer_[your model name]_[your name]."
	exit 1
fi
docker run --name $1 -it -d --net=host --shm-size=500g \
	--privileged=true \
	-w /home \
	--device=/dev/davinci_manager \
	--device=/dev/hisi_hdc \
	--device=/dev/devmm_svm \
	--entrypoint=bash \
	-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
	-v /usr/local/dcmi:/usr/local/dcmi \
	-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
	-v /usr/local/sbin/:/usr/local/sbin \
	-v /home:/home \
	-v /data:/data \
	acltransformer_glm130b:v1
