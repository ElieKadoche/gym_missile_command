# Reinforcement Learning

Here are some scripts to train and test Reinforcement Learning algorithms on the Missile Command environment.
For more details, see the `README.md` of each subfolder.

## Virtual environments

To avoid conflicts between different Python packages, you can use Python virtual environments.

## GPU support

To enable GPU support with PyTorch or TensorFlow, you need to have CUDA and cuDNN installed on your system.
You might encounter conflicts between versions.
If, for example, TensorFlow requires older versions of CUDA and cuDNN than the ones you have installed on your system, I recommend to use Docker containers.

Pleaser refer to this [link](https://docs.docker.com/engine/install/) to install Docker and this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#) to enable GPU support for Docker.
Then you can for example install the TensorFlow image with `sudo docker pull tensorflow/tensorflow:latest-gpu`.
