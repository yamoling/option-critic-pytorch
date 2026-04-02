FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04


ENV DEBIAN_FRONTEND=noninteractive

ARG TZ=Europe/Brussels
ARG userName=yamoling

# To install: 
RUN apt update -o Acquire::AllowInsecureRepositories=true
RUN apt upgrade -y --allow-unauthenticated
# Install common packages and open-cv requirements
RUN apt install -y tzdata sudo software-properties-common git bash-completion curl wget unzip zip psmisc htop ffmpeg libsm6 libxext6
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata


# Set root password to 'root'
RUN echo 'root:root' | chpasswd
# Create user in order to avoid running the container as root
#   -U to create a group with the same name as the user
#   -s to set the shell to bash
#   -p to set the password (hashed)
RUN useradd $userName -U --create-home -s /bin/bash -p $(openssl passwd -1 abc123)
# Add user to sudo group
RUN usermod -aG sudo $userName

# Enable bash completion
RUN echo 'if [ -f /etc/bash_completion ]; then' >> /etc/bash.bashrc \
    && echo '  . /etc/bash_completion' >> /etc/bash.bashrc \
    && echo 'fi' >> /etc/bash.bashrc

# Change user to avoid permission issues
USER ${userName}
RUN wget -qO- https://astral.sh/uv/install.sh | sh
# Add python package executables to path
ENV PATH="/home/${userName}/.local/bin:${PATH}"
ENV DEBIAN_FRONTEND=dialog

