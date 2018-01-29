nvcc = /share/apps/cuda-8.0/bin/nvcc
cudalib =  /share/apps/cuda-8.0/lib64
tensorflow = /home/yz6/.virtualenvs/gan_env/lib/python2.7/site-packages/tensorflow/include

TF_LIB= /home/yz6/.virtualenvs/gan_env/lib/python2.7/site-packages/tensorflow

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework  -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public  -lcudart -L $(cudalib) -L $(TF_LIB) -ltensorflow_framework  -O2 -D_GLIBCXX_USE_CXX11_ABI=0

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2



clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
