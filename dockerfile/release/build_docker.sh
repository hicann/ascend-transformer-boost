if [ ! -d "/data/tmp" ];
then
    mkdir "/data/tmp"
fi
cp -r /data/tmp .
docker build --no-cache --network=host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t acltransformer:release -f Dockerfile_release .
rm -rf tmp