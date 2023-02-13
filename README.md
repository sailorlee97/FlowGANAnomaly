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

@article{
  <br>
  li2022flowgananomaly, 
  <br>
  title={FlowGANAnomaly: Flow-based Anomaly Network Intrusion Detection with Adversarial Learning},
  <br>
  author={LI, Zeyi and WANG, Pan and WANG, Zixuan},
  <br>
  journal={Chinese Journal of Electronics},
  <br>
  volume={33},
  <br>
  pages={1--15},
  <br>
  year={2022},
  <br>
  publisher={Chinese Journal of Electronics}
}

