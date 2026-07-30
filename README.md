# solvent_screening_data
The datasets generated and/or analyzed during the current study, along with the source code for the machine learning models

## Dataset

- ER-100 dataset
  - ER-100 dataset is an EUV resist dataset, containing the molecular descriptor values of 100 resist molecules. 

- Sol-181 dataset
  - Sol-181 dataset is a solvent dataset, containing the molecular descriptor values of 181 solvent molecules.

- GDFE dataset
  - GDFE dataset comprises 18,100 samples (100 resist molecules × 181 solvent molecules) to train the CM-1 model.

- SC-P dataset
  - SC-P dataset comprises 300+ samples of solubility classification, collected from the published literature.

- SC-E dataset
  - SC-E dataset comprises 100+ samples of solubility classification, collected from the published literature.

## Model
the proposed ML framework to solvent screening in EUV resist systems

- environmental requirements

  - **python library**: pytorch and numpy

- input file
feature_input.txt: molecular descriptor values of solvent and resist, 40 columns

- command
```python
python predict.py
```

- output file
predict_result.txt
