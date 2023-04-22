if (Test-Path -Path .\data.tar) {
    if (-Not (Test-Path -Path .\data)){
        write "Extracting resources"
        tar -xvf .\data.tar
    }

    conda activate srm

    python main.py -m ablation -t_e 1
    python main.py

    cd ./visualizations
    python print_anchor_distribution.py
    cd ./hardness
    python cumulative_hardness.py -d 100
    cd ../..
}
else{
    write "Download the data.tar file and place in the repository root."
}
