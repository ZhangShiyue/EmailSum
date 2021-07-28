# EmailSum (ACL 2021)

This repository contains the data and code for the following paper:

[EmailSum: Abstractive Email Thread Summarization]()

```
@inproceedings{zhang2021emailsum,
  title={EmailSum: Abstractive Email Thread Summarization},
  author={Zhang, Shiyue and Celikyilmaz, Asli and Gao, Jianfeng and Bansal, Mohit},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Data

### Requirements

* Python 3
* requirements_data.txt
* Download Avocado Research Email Collection from [LDC](https://catalog.ldc.upenn.edu/LDC2015T03)

### Avocado
We collected the summaries of 2,549 Avocado email threads (see Avocado/summaries/EmailSum_data.json).
We collected one more reference for each of the 500 email threads in the testing set after submission 
(see Avocado/summaries/one_more_reference.json).

Avocado dataset's copyright is protected by Linguistic Data Consortium. So you need to first download 
Avocado from LDC and then prepare the data as below.

* First, cd Avocado/

* Download "emails.json" from [here](https://drive.google.com/file/d/1OK1fjBn269N3Cx8QUAFga1fWn9nNE_N8/view?usp=sharing) 
and put it under Avocado/

* Extract threads, assuming $ROOT_DIR contains the LDC2015T03 (i.e., $ROOT_DIR/LDC2015T03/Data/avocado-1.0.2)
```
python extract_threads.py --root_dir $ROOT_DIR
```
You will get "Avocado.json" which contains all extracted threads.


* Anonymize & Filter
```
python anonymize.py
```
After this step, you can see cleaned threads under "Avocado_threads/".


* Prepare Train/Dev/Test files
```
python standardize.py
```
After this step, you can see experimental files under "exp_data/".
There are two sub-directories: "data_email_short" and "data_email_long" for short
and long summary, respectively. 
Each line of the *.source file is one email thread, in which 
emails are separated by "|||".


### W3C
We provide the code for extracting threads from W3C email corpus for semi-supervised learning.

* First, cd W3C/

* Download raw data files from [here](https://drive.google.com/drive/folders/1ZPGdzvauoEN4qqsZ2ZxD4EkWobCZPZhZ?usp=sharing) 
and put them under W3C/raw_data/

* Extract threads
```
python extract_threads.py
```
You will get "W3C.json" which contains all extracted threads.

* Anonymize & Filter
```
python anonymize.py
```
After this step, you can see all cleaned thread under "W3C_threads/".


