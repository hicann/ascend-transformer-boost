if [ ! -d "/data/tmp" ];
then
    mkdir "/data/tmp"
fi
if [ ! -d "/data/acltransformer_testdata/weights/chatglm6b" ];
then
    mkdir -p "/data/acltransformer_testdata/weights/chatglm6b"
fi
cp -r /data/tmp .
cp -r /data/acltransformer_testdata/weights/chatglm6b .
docker build --no-cache --network=host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t acltransformer_chatglm6b:release -f Dockerfile_release .
rm -rf tmp
rm -rf chatglm6b
