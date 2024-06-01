# PScL-SDNNMAE
The code is the implementation of our method described in the paper “ Shenjian Gu, Matee Ullah, PScL-SDNNMAE: computational predictor for protein subcellular localization using classical and Masked Autoencoder-based multi-view features with ensemble feature selection approach”.
## (I) 1_FeatureExtractionCode
### (1)	data
There are two datasets:
#### (i)	Train dataset
The benchmark training dataset contains a total of 2,876 immunohistochemistry (IHC) images of seven different protein subcellular locations selected from the human protein atlas (HPA) database.
#### (ii)	Independent dataset
The independent dataset contains a total of 191 IHC images of seven different proteins selected from HPA. <br />
Please download the datasets from "https://drive.google.com/drive/folders/1L_ELeXRjq6cn6VzKtEfLKqWNJ4CcjokK?usp=sharing" and copy it to "data" folder.
### (2)	lib
lib folder contains all the necessary codes for classical handcraft features extraction used in this study.<br />
### (3)	mae_main
mae_main folder contains all the MAE-based deep features extraction related necessary codes used in this study.<br />
### (4)	Biomage_Feature_Extraction.m
Biomage_Feature_Extraction.m is the matlab file for extracting <br />
(1) Subcellular location features (Slfs) which includes
	(i)		DNA distribution features <br />
	(ii)	Haralick texture features <br />
(2)	Local binary pattern <br />
(3)	Completed local binary patterns <br />
(4)	Rotation invariant co-occurrence of adjacent LBP <br />
(5)	Locally encoded transform feature histogram and <br />
## (II)	2_FeatureSelectionCode
2_FeatureSelectionCode folder includes the following files.
### (1)	lib
lib folder contains all required files related to Analysis of Variance (ANOVA), Mutual Information (MI) and Stepwise discriminant analysis (SDA).
### (2) act_am.py
act_am.py is the python file which calls the ANOVA and MI feature selection algorithm.
### (3) act_sda.m
act_sda.m is the matlab file which calls the SDA feature selection algorithm.
### (4) data_concat_main.py
data_concat.py is the python file which integrate with the all the feature sets.
## (III)	3_ClassificationCode
3_ClassificationCode folder includes includes the following files.
### (1)	lib
lib folder contains all required files related to 4l-DNN and model evaluation.
### (2)	classification_main.py
classification_main.py is the python file which complete the classification task using the proposed method PScL-SDNNMAE.
## (IV)	Contact
If you are interested in our work or if you have any suggestions and questions about our research work, please contact us. E-mail: gushenjian@njust.edu.cn.
