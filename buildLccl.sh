set -e
rm -rf 3rdparty/lccl/include
rm -rf 3rdparty/lccl/lib
mkdir 3rdparty/lccl/include
mkdir 3rdparty/lccl/lib
cd Lccl2
rm -rf build
mkdir build
cd build
cmake -DASCEND_PATH=/home/wmj/Ascend/ascend-toolkit/latest ..
make -j lccl
sh ../build_cce.sh
cp lccl.o liblccl.so ../../3rdparty/lccl/lib/
cp lccl.o liblccl.so ../../../GLM-130B/
cp ../*.h ../../3rdparty/lccl/include/