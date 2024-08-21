#!/bin/bash

# either use conda or micromamba
mcba(){
    if command -v micromamba &>/dev/null; then
        micromamba "$@"
    else
        conda "$@"
    fi
}
mcba --version

mcba clean -a -y
mcba create -n confrank python=3.11.5
mcba activate confrank
mcba install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 lightning=2.1.1 torchmetrics=1.2.0 -c pytorch -c nvidia -c conda-forge
pip install torch-cluster==1.6.3 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --no-cache
pip install torch_geometric==2.5.0 h5py==3.10.0 seaborn==0.13.0 rdkit==2023.09.5 mace-torch==0.3.4 mlflow==2.9.1 black[d] numba pytest --no-cache
