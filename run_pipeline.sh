#!/bin/bash

export OMP_NUM_THREADS=1

if test -f "./data.tar" ; then
    if ! test -d "./data/" ; then
        echo "Extracting resources"
        tar -xvf ./data.tar
    fi

    # install R packages
    conda run --no-capture-output -n srm python preprocessing/r_packages.py

    conda run --no-capture-output -n srm python main.py -m ablation -t_e 1
    conda run --no-capture-output -n srm python main.py

    cd ./visualizations
    conda run --no-capture-output -n srm python print_anchor_distribution.py
    conda run --no-capture-output -n srm python cumulative_hardness.py -d 100
    cd ../

else
    echo "Download the data.tar file and place in the repository root."
fi
