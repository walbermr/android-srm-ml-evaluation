# Security Relevant Methods of Android’s API Classification: A Machine Learning Empirical Evaluation

To have all the sub-packages, run:

    $ git clone --recursive https://github.com/walbermr/android-srm-ml-evaluation.git

Install required packages:

    $ apt-get -y upgrade && apt-get install -y python3 wget git g++ gcc ca-certificates gnupg software-properties-common r-base
    
Install conda environment:

    $ conda env create -f env.yml

Install required R packages:

    $ conda run -n srm python ./r_packages.py

To replicate the results from the paper, you first must download the embedding algorithms checkpoints from the [gdrive link](https://drive.google.com/file/d/14-o2yLDIMDPg8NhQoGS2eJvtMux0YQr4/view?usp=share_link) and place it in this directory root. After extraction, it should have the following structure:

    ├── data
    │   ├── saved_models


To perform all experiments, run on Linux:

    $ ./run_pipeline.sh

On Windows, it is required to use the provided Docker image, as `Auto-sklearn` is not supported in Windows:

    $ ./docker-run.ps1
    
## Usage

To run SuSi, you just need to run the command below, if you get heap memory error, add -Xmx1024M . The objective is that SuSi will extract all the features needed to run the machine learning algorithms generating a static dataset instead of using the android.jar files to extract every time the classification is needed.

    $ java -cp lib/weka.jar:soot.jar:soot-infoflow.jar:soot-infoflow-android.jar:bin de.ecspride.sourcesinkfinder.SourceSinkFinder android-12.jar permissionMethodWithLabel.pscout out.pscout

To run on Windows use:

    $ java -cp "lib/weka.jar;soot.jar;soot-infoflow.jar;soot-infoflow-android.jar;bin" de.ecspride.sourcesinkfinder.SourceSinkFinder android-12.jar permissionMethodWithLabel.pscout out.pscout

To build SuSi you can use Eclipse directly or use:

    $ javac -d bin/ -cp lib/weka.jar:soot.jar:soot-infoflow.jar:soot-infoflow-android.jar -sourcepath src/ src/de/ecspride/sourcesinkfinder/SourceSinkFinder.java

To build on windows use:

    $ javac -d bin/ -cp "lib/weka.jar;soot.jar;soot-infoflow.jar;soot-infoflow-android.jar" -sourcepath src/ src/de/ecspride/sourcesinkfinder/SourceSinkFinder.java


## Troubleshooting

If you get heap memory error, add the flag `-Xmx1024M` while executing the java commands.
