FROM python:3.10

# Configure apt for setup
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install --no-install-recommends sudo \
    zsh \
    less \
    curl \
    wget \
    fonts-powerline \
    locales \
    graphviz
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

### GIT GLOBAL SETUP ###

RUN git config --global core.excludesfile /.globalgitignore
RUN touch /.globalgitignore
RUN echo ".notebooks" >> /.globalgitignore
RUN echo "nohup.out" >> /.globalgitignore

### ZSH TERMINAL SETUP ###

# generate locale for zsh terminal agnoster theme
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && /usr/sbin/locale-gen
RUN locale-gen en_US.UTF-8
# set term to be bash instead of sh
ENV TERM xterm
ENV SHELL /bin/zsh
# install oh-my-zsh
RUN sh -c "$(wget -nv -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

### PYTHON DEPENDENCIES INTALLATION ###

# upgrade pip and install package manager
RUN python -m pip install --no-cache-dir --upgrade pip

### TEAR DOWN IMAGE SETUP ###
# switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
