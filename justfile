init:
    git init

readme:
    python C:/Users/chaitrali/Documents/GitHub/readme-generator

commit message="init":
    git add .
    git commit -m {{message}}

cgan:
    conda activate w
    p src/RunGAN.py
    
test:
    echo "hi"

alias b := build

build:
  echo 'Building!'
