if [ $# -ne 1 ]; then
	echo "error: need one argument describing your proxy info."
	echo "usage: $0 [arg], arg in the format of http://..."
	exit 1
fi
docker build --no-cache --network=host --build-arg http_proxy=$1 --build-arg https_proxy=$http_proxy -t acltransformer_chatglm6b:v1 -f Dockerfile_dev .