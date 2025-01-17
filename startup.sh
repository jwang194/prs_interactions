conda env create -f breast_cancer.yaml
conda activate breast_cancer 

python3 -m ipykernel install --user --name=breast_cancer --display-name "breast_cancer"
jupyter notebook
