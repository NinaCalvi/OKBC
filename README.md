# OxKBC: Outcome Explanation for Factorization Based Knowledge Base Completion

Code accompanying AKBC'20 paper of the same title. Paper link: https://openreview.net/forum?id=nqYhFwaUj

We propose a method, that can be used to explain why an embedding based neural model trained for knowledge base completion gave that specific answer for a query. More specifically, given a knowledge base represented as a list of triples (e<sub>1</sub>, r, e<sub>2</sub>), we train an embedding based neural model to answer the query (e<sub>1</sub>, r, ?) which give the answer as u. We try to give reasoning for that answer for an arbitrary embedding based neural model. Presently, we are using a near state of the art model [TypeDM](https://github.com/dair-iitd/KBI/tree/master/kbi-pytorch)


## Getting Started

## Note
There is a slight change in template naming convention in this code.  
Templates 5 and 6 from the paper are templates 1 and 2 respectively in this repository.  
Templates 1, 2, 3 and 4 from the paper are templates 3, 4, 5 and 6 respectively in this repository.
Follow each section below to successfully replicate our results.

### Requirements

```
python >= 3.6
pytorch (version 0.3.1.post2)
numpy (version 1.13.3)
sklearn (version 0.19.1)
matplotlib (version 2.1.0)
pandas (version 0.20.3)
bs4 (beautiful soup 4 --> version 4.6.0)
pickle
argparse
```

### Project Structure
1. Clone this repository and `cd` into the project.
2. Run `mkdir dumps` and download this dump file here [https://drive.google.com/drive/folders/1kX3V9QRTPGSpF0c8Fh3LCoyFtxgM3pT6?usp=sharing](https://drive.google.com/drive/folders/1kX3V9QRTPGSpF0c8Fh3LCoyFtxgM3pT6?usp=sharing)
3. Download and extract the `fb15k` dataset into `data/fb15k` folder, such that you have `data/fb15k/train.txt`, `data/fb15k/valid.txt` and `data/fb15k/test.txt`. See [data section](data/README.md).

### Template Builder

We have a set of fixed templates, which we suppose are enough to explain some part of the dataset. We need an embedding dump of the neural model in the following format:
```
dump = {
    'relation_to_id': {relation name: int id},
    'entity_to_id': {entity name: int id},

    'entity_real': numpy.ndarray(shape=(number of entities, embedding dim),dtype=np.float32),
    'entity_type': numpy.ndarray(shape=(number of entities, type embedding dim),dtype=np.float32),

    'rel_real': numpy.ndarray(shape=(number of relations, embedding dim),dtype=np.float32),
    'head_rel_type': numpy.ndarray(shape=(number of relations, type embedding dim),dtype=np.float32),
    'tail_rel_type': numpy.ndarray(shape=(number of relations, type embedding dim),dtype=np.float32)
}
```
Make sure the embeddings are normalized, to do so use `scripts/normalize.py`, just change the file names. (If you completed the Project Structure section, you would already have the embeddings dump in `dumps` folder.

Now, we need to build template tables for these templates.

To do so, run the `template_builder.py` file

```
mkdir logs
mkdir logs/fb15k
python3 template_builder.py -h       ## Get help
python3 template_builder.py -d fb15k -m distmult -w dumps/fb15k_distmult_dump_norm.pkl -s logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ./data
```

This will save `1-6.pkl` in the save_directory which is `logs/fb15k` in the above example.


### Preprocessing

We need to preprocess the textual data to numeric data for our selection module as an input. To do so run the file `preprocessing.py` as given below:

```
## Generate a train file, where we do not have labels of y
python3 preprocessing.py -d fb15k -m distmult -f ./data/fb15k/train.txt -s logs/fb15k/sm_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ./data --negative_count 2

## Generate a valid/test file, where we do have labels of y
python3 preprocessing.py -d fb15k -m distmult -f ./data/fb15k/labelled_train/labelled_train_x.txt -s logs/fb15k/sm_valid_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ./data --negative_count 0 --y_labels ./data/fb15k/labelled_train/labelled_train_y6.txt
```

This will write a pkl and txt file with the name `logs/fb15k/sm_with_id.data.[pkl/txt]`. This file contains the data in the following format:

```
Column 0-2 contains the integer numeric id of e1, r and e2 in order
Column 3-44 contains the input vector for template 1,2,3,4,5,6 in order
Column 45 contains 1 if the fact is positive and 0 if it is negative(randomly sampled hence assumed false)
```

For the valid and test data generated with `--y_labels` flag, the column 45 contains the template id which produces the best explanation. It is annotated manually.

For this valid data, we shuffled it and randomly split into 80-20 ratio used as labelled train and valid data. The files generated were `sm_sup_train_with_id.pkl` and `sm_sup_valid_with_id.pkl` respectively, using the following script in `sm` folder:
```
python3 create_train_val_split.py --labelled_total_data_path ../logs/fb15k/sm_valid_with_id.data.pkl --total_labels_path ../data/fb15k/labelled_train/labelled_train_y6.txt --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt --train_split 0.8 --seed 242 --num_templates 6
```

### Training Selection Module
Next to train the selection module run the file `sm/main.py` as given below:

```
## Semi supervised training with KL Divergence = 0

python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train_semi_kl1 --num_epochs 20 --config ./configs/fb15k_config.yml --hidden_unit_list 90 40 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path ../logs/sm --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt

## Semi supervised training with KL Divergence = 1

python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train_semi_kl1 --num_epochs 20 --config ./configs/fb15k_config.yml --hidden_unit_list 90 40 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path ../logs/sm --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml  --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt

```
Refer to `sm/e2e.sh` for different settings of training.

This will save the best model (_best_checkpoint.pth0), a `learning_curve.txt` and a `log.txt` in the output folder named `[output_path]/[exp_name]`.

To test and evaluate, we use the `--only_eval` mode of the code

```

## To test the model, on a test file (generate it with preprocessing.py similar to valid data)

python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_sm.data.pkl --exp_name test_semi --num_epochs 20 --config ./configs/fb15k_semi.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path ../logs/sm/ --checkpoint ../logs/sm/train_semi/train_semi_r0.125_p1_n-2_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt

```
This will report the f score and log the predictions in `[output_path]/[exp_name]/[pred_file]`

Similarly test for KL divergence = 1.


### Amazon Mechanical Turk

We had two experiments on Amazon Mechanical Turk.

* Find out if our explanations are better than using rule mining. 
* Find out if explanations are infact useful for the task of KB verification. 

#### Is TeXKBC better?

We have an html file `turk_template.html` which is the form, which turkers have to fill. We need to generate a csv file containing information of HITs according to this HTML file. To do this use:

```
python3 get_turk_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k/ -o logs/fb15k/turk_better/ -tf ./data/fb15k/turk/test_hits1.txt -tp logs/sm/test_semi/preds.txt -rp [rule pred file] --data_repo_root ./data --num 5
```
It will generate a book file (which is a html file for easy viewing) and a `logs/fb15k/turk_better/turk_better_hits.csv` which is to be uploaded while creating a batch.

To analyze the results, download the `results.csv` file from MTurk, and run:

```
python3 get_turk_res.py -rf logs/fb15k/turk_better/results.csv -op logs/fb15k/turk_better/ -bf logs/fb15k/turk_better/turk_better_book.html
```

It then generates an analysis html file (`logs/fb15k/turk_better/results_analysis.html`) if all the HITs are valid, if Not it generates a CSV (`logs/fb15k/turk_better/results_rejected.csv`) with a reason for rejecting the HIT. Upload that CSV to MTurk to reject the HITs, not pay the turkers and republish the hits for other workers to do.


#### Are explanations useful?


We have an html file `turk_useful_template.html` which is the form, which turkers have to fill. We need to generate a csv file containing information of HITs according to this HTML file. To do this use:

```
python3 get_turk_useful_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k/ -o logs/fb15k/turk_useful/ -tf ./data/fb15k/turk/test_hits1.txt -tp logs/sm/test_semi/preds.txt --data_repo_root ./data --num 5
```

Essentially it generates a book file (which is a html file for easy viewing) and a `logs/fb15k/turk_useful/turk_useful_exps_hits.csv` and `logs/fb15k/turk_useful/turk_useful_no_exps_hits.csv`.
They contain the data for the facts, with explanation and without explanations respectively.

To analyze the results, download the `results.csv` file from MTurk, and run:

```
python3 get_turk_useful_res.py -rf logs/fb15k/turk_useful/results.csv -op logs/fb15k/turk_useful/
```

It then generates an analysis html file (`logs/fb15k/turk_useful/results_analysis.html`) if all the HITs are valid, if Not it generates a CSV (`logs/fb15k/turk_useful/results_rejected.csv`) with a reason for rejecting the HIT. Upload that CSV to MTurk to reject the HITs, not pay the turkers and republish the hits for other workers to do.

### Scripts

We had a number of scripts written to do small tasks, and it is present in the scripts directory. Most of them have their functionalities written at the top of the files as a doc string.

## Authors

* **Aman Agrawal** - [https://www.linkedin.com/in/aman71197/](https://www.linkedin.com/in/aman71197/)
* **Ankesh Gupta** - [https://www.linkedin.com/in/ankesh-gupta-a67423123](https://www.linkedin.com/in/ankesh-gupta-a67423123)
* **Yatin Nandwani** - [https://www.linkedin.com/in/yatin-nandwani-0804ba9/](https://www.linkedin.com/in/yatin-nandwani-0804ba9/)
* **Mayank Singh Chauhan** - [https://www.cse.iitd.ac.in/~cs5160394/](https://www.cse.iitd.ac.in/~cs5160394/)

See also the list of contributors who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## BibTeX Citation

If you use OxKBC in any publication, we would appreciate the following citation:

```
@inproceedings{
nandwani2020oxkbc,
title={Ox{\{}KBC{\}}: Outcome Explanation for Factorization Based Knowledge Base Completion},
author={Yatin Nandwani and Ankesh Gupta and Aman Agrawal and Mayank Singh Chauhan and Parag Singla and Mausam},
booktitle={Automated Knowledge Base Construction},
year={2020},
url={https://openreview.net/forum?id=nqYhFwaUj}
}
```

## Acknowledgments

Project completed under the guidance of

* **Mausam** - [http://www.cse.iitd.ac.in/~mausam/](http://www.cse.iitd.ac.in/~mausam/)
* **Parag Singla** - [http://www.cse.iitd.ac.in/~parags/](http://www.cse.iitd.ac.in/~parags/)
