# Get the base image from Docker Hub
FROM osrf/ros:foxy-desktop-full

WORKDIR /robotics

# Update apps on the base image
# RUN rm /etc/apt/sources.list.d/ros2-latest.list && \
#     apt update && apt install curl && \
#     curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update
RUN  apt-get install -y curl  
RUN  apt-get install -y lsb-release 
RUN  apt-get install -y gnupg 
RUN  apt-get install -y tmux 
RUN  apt-get install -y  vim 
RUN  apt-get install -y  wget
RUN  apt-get install -y  python3-pip 

