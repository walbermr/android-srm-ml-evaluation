if [ test -f ./data.tar ]; then
    if [! test -d ./data ]; then
        echo "Extracting resources"
        tar -xvf ./data.tar
    fi

    conda activate srm

    python main.py -m ablation -t_e 1
    python main.py

    cd ./visualizations
    python print_anchor_distribution.py
    python cumulative_hardness.py -d 100
    cd ../

else
    echo "Download the data.tar file and place in the repository root."
fi
