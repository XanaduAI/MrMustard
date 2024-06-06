#! /bin/sh

apt-get update -y
apt-get -y -o Dpkg::Options::="--force-confold" --force-yes install --no-install-recommends zsh
apt-get -y install --no-install-recommends fonts-powerline locales toilet fortunes fortune-mod
apt-get clean && rm -rf /var/lib/apt/lists/*

# install jupyter notebook widgets extension
pip install ipywidgets ipykernel

# generate locale for zsh terminal agnoster theme
echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && /usr/sbin/locale-gen
locale-gen en_US.UTF-8

# *** Install oh-my-zsh ***
# Make zsh the default shell
chsh -s $(which zsh)
# Disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
git config --add oh-my-zsh.hide-dirty 1
# Download  and run installation script
curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | sh

# Print welcome message
clear
echo "\n"
/usr/games/fortune -w wisdom tao science songs-poems literature linuxcookie linux education art
echo "\n"
toilet --metal -f emboss -W MrMustard
echo "\n"
echo "This devcontainer is ready for development!"
