# Precise Target-Oriented Attack against Deep Hashing-based Retrieval
This is the code for our MM 2023 submission "[Precise Target-Oriented Attack against Deep Hashing-based Retrieval](https://openreview.net/forum?id=G_D5JQCQW2h&referrer=%5BTasks%5D(%2Ftasks))"

## Usage
#### Dependencies
- Python 3.7.16
- Pytorch 1.10.0
- Numpy 1.18.5
- CUDA 11.3

#### Hashing models training
For Deep Hashing models training, please refer to https://github.com/swuxyj/DeepHash-pytorch. 
We provide the model files of CSQ on NUS-WIDE, FLICKR-25K, and MS-COCO. To run our main experiments, please download them directly from https://drive.google.com/drive/folders/10qlmgzZSIRDmhItwVji3CZnPxq2-VkBC?usp=sharing, the default directory for storing the model files is /model_weights.


#### Evaluate PTA
Download the necessary files from this anonymous link (https://drive.google.com/drive/folders/1dLW_chQRh6npW6uW8-5iCroml7R-tXI9?usp=sharing) and place them in /log. Also download the standard datasets (NUS-WIDE, FLICKR-25K, MS-COCO)  and set their path in arguments **data_path**.
Initialize the hyper-parameters in main.py following the paper, here are some useful arguments:

- --multi | True:single-target label selection and False:general-target label selection
- --attack | True:in-classes-case and False:out-of-classes case
- --cuda | choose device

For example, to run our PTA on NUS-WIDE under out-of-classes case using 32 bits hash code:
```
python main.py --data 'NUS-WIDE' --multi False --attack False --bit 32 --cuda 0
```
Or, to run our PTA on MS-COCO under in-classes case using 64 bits hash code:
```
python main.py --data 'MS-COCO' --multi False --attack True --bit 64 --cuda 0
```

