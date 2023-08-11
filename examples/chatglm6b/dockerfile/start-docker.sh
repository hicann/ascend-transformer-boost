# Usage:
# 1. Configure proxy in your host machine.
# 2. bash build_docker.sh [your proxy(http://...)]
# 3. start_docker.sh [your container name(acltransformer_chatglm6b_[your name])]
# 4. docker exec -it --user root [your container name] bash
# 5. configure /home/proxy.sh if you need to have access to Internet

if [ $# -ne 1 ]; then
	echo "error: need one argument describing your container name."
	echo "usage: $0 [arg], arg in the format of acltransformer_chatglm6b_[your name]."
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
	-v /usr/local/asdops:/usr/local/asdops \
	-v /data/acltransformer_testdata:/data/acltransformer_testdata \
	acltransformer_chatglm6b:v1
