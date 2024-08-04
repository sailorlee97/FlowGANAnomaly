# FlowGANAnomaly
This project is an open-source project based on a GAN network anomaly detection ‘zero day’ attack.

### Structure
`process_nslkdd.py`: Load the malicious traffic data set that needs to be trained

` model.py`: Model building and training process

` networks.py`:  Basic network structure, it is not recommended to change easily

`option.py`: Parameter file, this file supports four data sets: UNSW-NB15, CIC-IDS2017, NSL-KDD and CIC-DDoS2019

### How to use?

Choose a dataset for training, default NSL-KDD ,and then run ` python train.py`

if you want to change dataset, you need  change `--dataset` and ` --feature` parameters.

### References  
datasets are from CICIDS2017(https://www.unb.ca/cic/datasets/ids-2017.html),  
UNSW-NB15(https://research.unsw.edu.au/projects/unsw-nb15-dataset),   
CIC-DDoS2019(https://www.unb.ca/cic/datasets/ddos-2019.html)

### Citation
If you find this useful in your research, please consider citing:
```
@article{li2024flowgananomaly,
  title={FlowGANAnomaly: Flow-based Anomaly Network Intrusion Detection with Adversarial Learning},
  author={LI, Zeyi and WANG, Pan and WANG, Zixuan},
  journal={Chinese Journal of Electronics},
  volume={33},
  number={1},
  pages={1--14},
  year={2024},
  publisher={Chinese Journal of Electronics}
}
```

### Acknowledgement
This repository adapts the code published under the [Semi-supervised Anomaly Detection via Adversarial Training](https://github.com/samet-akcay/ganomaly) repository. Please also consider citing the following if you re-use the code:
```
@inproceedings{akcay2018ganomaly,
  title={Ganomaly: Semi-supervised anomaly detection via adversarial training},
  author={Akcay, Samet and Atapour-Abarghouei, Amir and Breckon, Toby P},
  booktitle={Asian Conference on Computer Vision},
  pages={622--637},
  year={2018},
  organization={Springer}
}
```
## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIDC-AI/Parrot&type=Date)](https://star-history.com/#AIDC-AI/Parrot&Date)

### Funding
The paper is sponsored by National Natural Science Fundation (General Program) Grant 61972211, China,National Key Research and Development Project Grant 2020YFB1804700, China, and Future Network Innovation Research and Application Projects No.2021FNA02006.
