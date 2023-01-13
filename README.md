# FlowGANAnomaly
This project is an open-source project based on a GAN network anomaly detection ‘zero day’ attack.

### Structure
`process_nslkdd.py`: Load the malicious traffic data set that needs to be trained

` model.py`: Model building and training process

` networks.py`:  Basic network structure, it is not recommended to change easily

`option.py`: Parameter file, this file supports three data sets: UNSW-NB15, CIC-IDS2017, NSL-KDD and CIC-DDoS2019

### How to use?

Choose a dataset for training, default NSL-KDD ,and then run ` python train.py`

if you want to change dataset, you need  change `--dataset` and ` --feature` parameters.

### References

datasets are from CICIDS2017(https://www.unb.ca/cic/datasets/ids-2017.html) , 
UNSW-NB15(https://research.unsw.edu.au/projects/unsw-nb15-dataset), 
CIC-DDoS2019(https://www.unb.ca/cic/datasets/ddos-2019.html)

### Citation

If you find this useful in your research, please consider citing:

@articleInfo{E220173,
title = "FlowGANAnomaly: Flow-based Anomaly Network Intrusion Detection with Adversarial Learning",
journal = "Chinese Journal of Electronics",
volume = "33",
number = "E220173,
pages = "1",
year = "2022",
note = "",
issn = "",
doi = "10.23919/cje.2022.00.173",
url = "https://cje.ejournal.org.cn/en/article/doi/10.23919/cje.2022.00.173",
author = "LI Zeyi","WANG Pan","WANG Zixuan",keywords = "Anomaly detection","Unsupervised learning","Generative adversarial network","Intrusion detection system"}

