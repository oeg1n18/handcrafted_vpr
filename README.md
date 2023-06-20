# Handcrafted VPR Template Code

This small repository gives implementation of the following VPR techniques 
1. HOG - Histogram of gradients 
2. SIFT-BOW - SIFT features with a Bag of Visual Words Aggregation 
3. Cohog

It also provides functionality to download and use the following datasets 
1. SFU
2. GardensPointWalking
3. StLucia 

Please see the code in 'main.py' and read the comments to see how to use the methods. 

To Run this repository complete the following. First is to create a virtual environment with all the required packages. It is recommended this is achieved with conda. To install miniconda on your machine see https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html. 

Alternatively you can install the required packages using anaconda navaigator. These packages can be seen in the 'requirements.txt' file.

```bash
# Create an environment.Run the following in bash 
conda create --name handcrafted_vpr_env
conda activate handcrafted_vpr_env
conda install python
# install all the packages 
pip install -r requirements.txt
```

It is also very important that you set the root directory path in the 'config.py' file. 
This will bet the absolute directory path to the repository root for example it will be '/home/path/to/VPR_handcrafted'
