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
* requirements.txt
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

* First, cd "W3C/"

* Download raw data files from [here](https://drive.google.com/drive/folders/1ZPGdzvauoEN4qqsZ2ZxD4EkWobCZPZhZ?usp=sharing) 
and put them under "W3C/raw_data/"

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


## Model

### Requirements

* Python 3
* PyTorch 1.7, transformers==2.11.0

### Test pre-trained models

* Download pre-trained models from [here](https://drive.google.com/drive/folders/1KXnoGpyzcwESfLN9JmzM0gDz_75WVkaj?usp=sharing), decompress, and put them under "train/".

Note that we conduct model selection for each metric, so there are multiple 
best checkpoints, e.g., "checkpoint-rouge1" is the best ROUGE1 checkpoint selected by ROUGE1 on 
development set. "best_ckpt.json" contains the best scores on development set. 

* Prepare data

After you get "Avocado/exp_data/data_email_short" and "Avocado/exp_data/data_email_long", run 
```
python3 data.py --data_dir Avocado/exp_data/data_email_long --cache_dir train/cache --max_output_length 128  
python3 data.py --data_dir Avocado/exp_data/data_email_short --cache_dir train/cache --max_output_length 56  

```

* Test

T5 baselines
```
python3 run.py --task email_long --data_dir Avocado/exp_data/data_email_long/ --test_only --max_output_length 128
python3 run.py --task email_short --data_dir Avocado/exp_data/data_email_short/ --test_only --max_output_length 56
```

Hierarchical T5
```
python3 run.py --task email_long --memory_type ht5 --data_dir Avocado/exp_data/data_email_long/ --test_only --max_output_length 128
python3 run.py --task email_short --memory_type ht5 --data_dir Avocado/exp_data/data_email_short/ --test_only --max_output_length 56
```

Semi-supervised models
```
python3 run.py --task email_long_w3c --data_dir Avocado/exp_data/data_email_long/ --test_only --max_output_length 128
python3 run.py --task email_short_together --data_dir Avocado/exp_data/data_email_short/ --test_only --max_output_length 56
```

The testing scores will be saved in "best_ckpt_test.json". 
We provide "best_ckpt_test_verification.json" for verification of results, almost the same numbers should be obtained.

We also provide "best_ckpt_test_old.json" that contains our previously tested scores (reported in the paper).
You are likely to get slightly different numbers from "best_ckpt_test_old.json" because we added a few more data
clean and anonymization rules. The pre-processed *.source files will be 
slightly different from the ones we used before.  

* Test with two references

Just add "--two_ref", e.g.,
```
python3 run.py --task email_long --data_dir Avocado/exp_data/data_email_long/ --test_only --two_ref --max_output_length 128
```

The testing scores will be saved in "best_ckpt_test_2ref.json". 
We provide "best_ckpt_test_2ref_verification.json" for verification of results, almost the same numbers should be obtained.


### Benchmark Results
**One-reference** results:

| EmailSum **Short** | rouge1       | rouge2       | rougeL       | rougeLsum    | BERTScore    |
| :------------- | :----------: | -----------: | -----------: | -----------: | -----------: |
|  T5 base       | 36.61        | 10.58        | 28.29        | 32.77        | 33.92        |
|  HT5           | 36.30        | 10.74        | 28.52        | 33.33        | 33.49        |
|  Semi-sup. (together)| 36.99  | 11.22        | 28.71        | 33.70        | 33.91        |

| EmailSum **Long** | rouge1       | rouge2       | rougeL       | rougeLsum    | BERTScore    |
| :------------- | :----------: | -----------: | -----------: | -----------: | -----------: |
|  T5 base       | 43.87        | 14.10        | 30.50        | 39.91        | 32.07        |
|  HT5           | 44.44        | 14.51        | 30.86        | 40.24        | 32.31        |
|  Semi-sup. (w3c)| 44.58       | 14.64        | 31.40        | 40.73        | 32.80        |


**Two-reference** results (average the results of two references):

| EmailSum **Short** | rouge1       | rouge2       | rougeL       | rougeLsum    | BERTScore    |
| :------------- | :----------: | -----------: | -----------: | -----------: | -----------: |
|  T5 base       | 35.22        | 9.60         | 27.08        | 31.22        | 32.45    |
|  HT5           | 34.81        | 9.82         | 27.28        | 31.74        | 32.42        |
|  Semi-sup. (together)| 35.52  | 10.35        | 27.29        | 33.11        | 32.24        |

| EmailSum **Long** | rouge1       | rouge2       | rougeL       | rougeLsum    | BERTScore    |
| :------------- | :----------: | -----------: | -----------: | -----------: | -----------: |
|  T5 base       | 43.41        | 13.81        | 29.97        | 39.32        | 31.58        |
|  HT5           | 43.86        | 14.06        | 30.17        | 39.64        | 31.84        |
|  Semi-sup. (w3c)| 43.99       | 14.18        | 30.56        | 40.12        | 32.04        |

Interestingly, we always get lower scores when comparing to the 2nd reference we collected after 
paper submission. That's why two-reference results are always worse than one-reference ones. 
It may be caused by the different set of turkers involved in summary annotation that 
brings domain shift.

### Train

Just drop "--test_only", e.g.,
```
python3 run.py --task email_long --data_dir Avocado/exp_data/data_email_long/ --max_output_length 128
```




