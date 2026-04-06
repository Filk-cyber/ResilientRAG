## data
Please download the data/directory data from [the link](https://drive.google.com/file/d/1CeQDiLggq3zlb4924RBu5pI4FuX3RcJV/view?usp=drive_link).
## Preprocessing
After downloading the dataset, run the following command to create the development and test sets used in our experiments:
```python
python preprocessing.py \
    --dataset hotpotqa \
    --raw_data_folder data/hotpotqa/raw_data \
    --save_data_folder data/hotpotqa 
```
Among them, `--raw_data_folder` specifies the folder containing the raw data, and `--save_data_folder` specifies the folder where development and testing data will be saved.
## Evaluation
HAMMER needs to go through the following four steps to evaluate performance.
If you are only interested in the final results, you can directly use the final test dataset `hotpotqa_test1000ideal_with_reasoning_chains_fakenum1_linear_w0.4.json` in step 4 and evaluate performance by running the provided commands.
### Step 1
First, run `add_wronganswer.py` to obtain the error answer, then run `add_orifake.py` to obtain the misinformation, and finally `run addCtxs.py`  to format the error information obtained.
### Step 2
Run the followiing command to generate KGs:
```python
python generate_knowledge_triples.py \
    --dataset hotpotqa \
    --input_data_file data/hotpotqa/test.json \
    --save_data_file data/hotpotqa/test_with_kgs.json
```
Then run `add_truthful_scores.py` to obtain the credibility score for each triple.
### Step 3
Run the following command to construct hybrid-scored reasoning chains:
```python
python construct_reasoning_chains.py \
  --dataset hotpotqa \
  --input_data_file data/hotpotqa/test_add_scores_with_kgs.json \
  --output_data_file data/hotpotqa/test_with_reasoning_chains.json \
  --calculate_ranked_prompt_indices \
  --fake_num 1 \
  --scoring_function linear\
  --max_chain_length 4 
```
### Step 4
Run the following command to evaluate the QA performance:
```python
python evaluation.py \
  --test_file data/hotpotqa/test_with_reasoning_chains.json \
  --reader llama3 \
  --context_type triples \
  --n_context 5 
```
