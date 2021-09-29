# Summarization
This is the code for Mitacs GRI 2021.

## Citation:
Some of the code for this project is based on the paper "[Multilevel Text Alignment with Cross-Document Attention](https://xuhuizhou.github.io/Multilevel-Text-Alignment/)"

## Installation: 
```bash
conda env create -f environment.yml
conda activate dev
```

If the above method does not work, you can use the following versions of the packages:
```
pip install torch==1.5.0
pip install transformers==2.3.0
```

## Example commands
First, download the [dataset](https://xuhuizhou.github.io/Multilevel-Text-Alignment/) (Please always download the latest version).

You need to obtain the pre-trained contextualized embedding (.npy file as well as .index file) first to run the code. Though there are many ways to achieve that, we recommend using the following command (you can find get_rep.py in this repo): 

```
!python ./get_rep.py \
    --output_dir=cite_models \
    --overwrite_output_dir \
    --model_type=bert \
    --per_gpu_eval_batch_size=10 \
    --model_name_or_path=bert-large-cased \
    --line_by_line \
    --train_data_file=./dev_cite.txt or path_to_your_txt_file \
    --special_eval \
    --eval_data_file=./dev_cite.txt or path_to_your_txt_file\
    --rep_name=./dev_cite.npy or destination_path\
    --mlm
```

To train the model please run the following command:
```
!python ./train_bert_g.py \
    --lr=0.00001 \
    --batch_size=256 \
    --hidden_size=50 \
    --max_len=10 \
    --train_set=./dev.csv \
    --test_set=./test.csv \
    --saved_path=./ \
    --model_type=Bert_han_sg_g
```

To test the model, run:
```
!python ./CDA/test_bert_g.py \
    --data_path="dev.csv" \
    --pre_trained_model="/Bert_han_sg_g.pth" \
    --max_len=10 \
    --model_type='Bert_han_sg_g' 
```

## Important files
You can find the modified attention and the dataloader in the src folder in the file called MUSEAttention.py and dataset_bert_4.py respectively. 


