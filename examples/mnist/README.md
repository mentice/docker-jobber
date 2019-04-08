## Introduction
This example illustrates separating data generation from ML training by creating two jobber images - one containing just the training data and another containing training code.
This workflow permits sharing of data between training runs, reduces the size of the generated docker images, and shortens the training cycle.


### Data Image

We'll download the mnist data set using a simple Python script executed during the build process. 
This is a common idiom, but we could further decompose data generation by creating an image with just the code for data generation, and run that image to produce yet another image containing the training data. Arbitrarily complex pipelines may be constructed in this manner.

Here's the code to download the mnist data set. The data is exported as a numpy array to the `/data` directory:

```Python
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

mnist_data = mnist.load_data()

np.save('/data/mnist.npy', mnist_data)
```

And the Dockerfile that runs the script during the build process:

```Dockerfile
# Build:
#   jobber build -f Dockerfile_data -t mnist-data 

FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /work
RUN mkdir /data
COPY mnist_data.py .
RUN python mnist_data.py
CMD echo "nothing to do"
```

The final echo `CMD` is used to override the default behavior from the tensorflow image which opens a Jupyter notebook.

We're now ready to build the data image:

```
$ jobber build -f Dockerfile_data -t mnist-data
$ docker image list mnist-data
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mnist-data          20190309_192158     40695c224698        50 seconds ago      3.14GB
mnist-data          latest              40695c224698        50 seconds ago      3.14GB
```

Notice that the mnist-data image has two tags: 'latest' and a UTC time-stamp (YYYMMDD_HHMMSS). Docker Jobber always produces a uniquely identifyable image - everything is captured.

Also, don't worry too much about the image size (3.14GB).  Docker stores images as a series of layers:

```
$ docker image history mnist-data:latest
IMAGE               CREATED              CREATED BY                                      SIZE                COMMENT
40695c224698        About a minute ago   /bin/sh -c #(nop)  LABEL jobber.version=0.1.0   0B                  
e77fcc075f61        About a minute ago   /bin/sh -c #(nop)  LABEL jobber.build-tags=[…   0B                  
8a01eba36dcd        15 minutes ago       /bin/sh -c #(nop)  CMD ["/bin/sh" "-c" "echo…   0B                  
69c96f0c34ce        19 minutes ago       /bin/sh -c python mnist_data.py                 73.7MB              
8588ca109ff3        About an hour ago    /bin/sh -c #(nop) COPY file:b650c0285f05bc81…   147B                
7c7475d752b9        About an hour ago    /bin/sh -c mkdir /data                          32.2kB              
61a6daf95f5c        4 months ago         /bin/sh -c #(nop) WORKDIR /work                 0B                  
6243acd2b19f        6 months ago         /bin/sh -c #(nop)  CMD ["/run_jupyter.sh" "-…   0B                  
<missing>           6 months ago         /bin/sh -c #(nop) WORKDIR /notebooks            0B                  
<missing>           6 months ago         /bin/sh -c #(nop)  EXPOSE 8888                  0B                  
<missing>           6 months ago         /bin/sh -c #(nop)  EXPOSE 6006                  0B                  
<missing>           6 months ago         /bin/sh -c #(nop)  ENV LD_LIBRARY_PATH=/usr/…   0B                  
<missing>           6 months ago         /bin/sh -c #(nop) COPY file:523b42812b104f41…   733B                
<missing>           6 months ago         /bin/sh -c #(nop) COPY dir:4679761a8b21dd519…   400kB               
<missing>           6 months ago         /bin/sh -c #(nop) COPY file:50103defeea39e9f…   1.16kB              
<missing>           6 months ago         /bin/sh -c ln -s -f /usr/bin/python3 /usr/bi…   16B                 
<missing>           6 months ago         /bin/sh -c pip3 --no-cache-dir install /tens…   987MB               
<missing>           6 months ago         /bin/sh -c #(nop) COPY file:ab6bdeecb6c7f183…   253MB               
<missing>           6 months ago         /bin/sh -c pip3 --no-cache-dir install      …   365MB               
<missing>           6 months ago         /bin/sh -c curl -O https://bootstrap.pypa.io…   12.4MB              
<missing>           6 months ago         /bin/sh -c apt-get update && apt-get install…   1.31GB              
<missing>           6 months ago         /bin/sh -c #(nop)  LABEL maintainer=Craig Ci…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV NVIDIA_REQUIRE_CUDA=c…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV NVIDIA_DRIVER_CAPABIL…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV NVIDIA_VISIBLE_DEVICE…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV LD_LIBRARY_PATH=/usr/…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV PATH=/usr/local/nvidi…   0B                  
<missing>           7 months ago         /bin/sh -c echo "/usr/local/nvidia/lib" >> /…   46B                 
<missing>           7 months ago         /bin/sh -c #(nop)  LABEL com.nvidia.cuda.ver…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  LABEL com.nvidia.volumes.…   0B                  
<missing>           7 months ago         /bin/sh -c apt-get update && apt-get install…   1.53MB              
<missing>           7 months ago         /bin/sh -c #(nop)  ENV CUDA_PKG_VERSION=9-0=…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  ENV CUDA_VERSION=9.0.176     0B                  
<missing>           7 months ago         /bin/sh -c apt-get update && apt-get install…   16.1MB              
<missing>           7 months ago         /bin/sh -c #(nop)  LABEL maintainer=NVIDIA C…   0B                  
<missing>           7 months ago         /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B                  
<missing>           7 months ago         /bin/sh -c mkdir -p /run/systemd && echo 'do…   7B                  
<missing>           7 months ago         /bin/sh -c sed -i 's/^#\s*\(deb.*universe\)$…   2.76kB              
<missing>           7 months ago         /bin/sh -c rm -rf /var/lib/apt/lists/*          0B                  
<missing>           7 months ago         /bin/sh -c set -xe   && echo '#!/bin/sh' > /…   745B                
<missing>           7 months ago         /bin/sh -c #(nop) ADD file:204fb7ccb19ff7e86…   115MB 

```

Only the top seven layers (about 74Mb) actually take up additional disk space - the rest comes from the tensorflow image.


### Code Image

Now we're ready to build the training code image. 
Here's the Tensorflow code (notice that `mnist.npy` is loaded from the `/data` directory which will be mounted when the image is run):

```Python
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

(x_train, y_train),(x_test, y_test) = np.load('/data/mnist.npy')
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2)
loss, metrics = model.evaluate(x_test, y_test)
print('loss={0} accuracy={1}'.format(loss, metrics))
```

And the Dockerfile:

```Dockerfile
# Build:
#   jobber build -f Dockerfile_train       # Specify Dockerfile name - defaults to cwd (mnist)
# Run:
#   jobber run -i mnist-data:latest-run mnist

FROM tensorflow/tensorflow:latest-gpu-py3
# FROM tensorflow/tensorflow:latest-py3

WORKDIR /work
COPY mnist.py .

CMD python mnist.py
```

Build and run:

```
$ jobber build -f Dockerfile_train
$ jobber run -i mnist-data mnist
Epoch 1/2
2019-03-09 19:41:03.246539: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-03-09 19:41:03.334695: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:897] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-09 19:41:03.335091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1405] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 10.92GiB freeMemory: 10.46GiB
2019-03-09 19:41:03.335109: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1484] Adding visible gpu devices: 0
2019-03-09 19:41:03.510157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:965] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 19:41:03.510185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971]      0 
2019-03-09 19:41:03.510191: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] 0:   N 
2019-03-09 19:41:03.510367: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1097] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10115 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
60000/60000 [==============================] - 4s 59us/step - loss: 0.2201 - acc: 0.9346
Epoch 2/2
60000/60000 [==============================] - 3s 42us/step - loss: 0.0985 - acc: 0.9697
10000/10000 [==============================] - 0s 16us/step
loss=0.08564752679057419 accuracy=0.9723
```

The mnist-data image is referred to on the command line using the `-i` switch. Input data images are automatically mounted as volumes in the executing image's file system. The default mount location is `/data`, but this may be overridden (e.g. `-i mnist-data,src=/foo`).

Running the training image produced yet another output image (tagged as `latest-run`) capturing the state of the file system:

```
$ docker image list mnist
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
mnist               20190309_195119     f245ce455242        12 seconds ago      3.07GB
mnist               latest-run          f245ce455242        12 seconds ago      3.07GB
```

The result image could be used to restart training from snapshot files saved during the training run, or for post-processing etc.