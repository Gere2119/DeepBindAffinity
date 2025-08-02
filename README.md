# DeepBindAffinity
DeepBindAffinity is a deep learning methd based on ResNet and a multihead-attention mechanism for predicting protein-ligand binding affinity.

The raw data can be found at [PDBbind](https://www.pdbbind-plus.org.cn/) database.

Data preprocessing can be referred to ./precode/. The processed data is stored in ./data/.
# Requirements

python 3.9.12

numpy 1.21.5

sklearn 1.1.2

tqdm 4.64.0

numba 0.55.1

Cuda 11.4.100
pytorch 1.12.1

pandas 1.4.3

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Gere2119/DeepBindAffinity
2. Navigate to the project folder:
   ```bash
   cd DeepBindAffinity
   
You can create GPU enabed environment
    ```bash
    
    conda env create -f environment.yml
    conda activate pytorch

## Training & Testing

  Navigate to src folder 
  
       cd ./src/
  To train the model
     
    python main.py

To train the model

    python test.py


## Evaluation Metrics

The model outputs five key performance indicators after running the test script:
    
     - **RMSE** (Root Mean Square Error)
     - **MAE** (Mean Absolute Error)  
     - **SD** (Standard Deviation)  
     - **CI** (Confidence Interval)  
     - **R** (Correlation Coefficient)

  
The training result is stored in `./src/Results/kfold-result.csv`
The test result is stored in `./src/Results/test-result.csv`

Every output file follows the same structure with three data columns: protein identifiers in the first column, measured binding affinities in the second column, and our model's predicted affinities in the third column.

## Contact

For questions, feedback, or collaborations:

- **Email**: [224718025@csu.edu.cn](224718025@csu.edu.cn)  or  [ georgewediasse21believer@gmail.com]( georgewediasse21believer@gmail.com)
- **GitHub Issues**: [Open an issue](https://github.com/Gere2119/DeepBindAffinity/issues)   

*We welcome contributions and suggestions!*
