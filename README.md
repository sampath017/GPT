# GPT

## jarvis labs startup script
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install python3.12 nvtop htop -y
apt-get remove python3 -y
alias python='python3.12'
apt-get install -y python3.12-dev
apt-get install python3-pip -y

pip install poetry
export PATH="$HOME/.local/bin:$PATH"
poetry config virtualenvs.in-project true

git config --global user.email "118274231+sampath017@users.noreply.github.com"
git config --global user.name "sampath"
git config --global init.defaultBranch "main"
git config --global push.default "simple"
git config --global pull.default "current"
git config --global credential.helper "store"

cd /home
rm -rf GPT
git clone https://github.com/sampath017/GPT.git
cd GPT
poetry install

# Prime intellect startup script
apt-get update
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install htop git nvtop python3-pip

pip install uv

git config --global user.email "118274231+sampath017@users.noreply.github.com"
git config --global user.name "sampath"
git config --global init.defaultBranch "main"
git config --global push.default "simple"
git config --global pull.default "current"
git config --global credential.helper "store"
git config pull.rebase false

cd /workspace
rm -rf GPT
git clone https://github.com/sampath017/GPT.git
cd GPT
poetry install
