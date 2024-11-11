# Installation

Install the latest Prom release using:

```
pip install git+https://github.com/HuantWang/PROM.git
```

PROM was tested with Python 3.7 to 3.9 and Ubuntu 18.04.

## DOCKER
Follow the instructions below to use our AE evaluation scripts.

### 1. Setup

Install Docker by following the instructions 
[here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). 
The following instructions assume the host OS runs Linux.

#### 1.1  Fetch the Docker Image

Fetch the docker image from docker hub.

```
$ sudo docker pull wanghuanting/prom:0.1
```

To check the list of images, run:

```
$ sudo docker images
#output
#REPOSITORY                                                               TAG                                 IMAGE ID       CREATED         SIZE
#wanghuanting/prom                                                        0.1                                cc84e8929fe1   2 minutes ago    147GB

```

Run the Docker container

```
$ sudo docker run -it --name prom -p 8099:8099  wanghuanting/prom:0.1 /bin/bash
```


#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate thread
``````

Then, go to the directory of our tool:

```
# Move the project to the target directory
(thread) $ mv /cgo/PROM/prom/* /cgo/prom/
(thread) $ cd prom/PROM/examples/tutorial/
```
