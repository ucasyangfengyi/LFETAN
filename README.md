## Requirements

* pytorch >= 1.4.0
* transformers >= 3.3.1

## Data
We evaluate our proposed model on the benchmark slot tagging dataset SNIPS and the name entity recognition dataset NER, 
followed the same data split provided by (Hou et al. 2020).

We employ SpaCy’s pre-trained POS tagger to mark POS labels.
The suffix of the data file name containing POS tags is 'fullfeature'.
All data is contained in the 'data' folder.

## Important code files
slot_tagging_with_prototypical_network_with_pure_bert.py    ## Training and Testing strategy
slot_intent_with_prototypical_network_and_pure_bert.py      ## Model structure
FusionNet.py    ## Knowledge fusion
crf.py          ## Calculate loss

Please note that these files are the second phase code.
The basic model training in the first phase requires simple modification, and then reuses these codes.
You need to comment out the relevant code of POS model and FusionNet.

## Training & Validation & Evaluation
```shell 
bash run_few_shot_slot_tagger_protoNet_with_pure_bert.sh \
    --matching_similarity_y ctx \
    --matching_similarity_type xy1 \
    --matching_similarity_function dot \
    --test_finetune false \
    --random_seed 999 \
    --model_removed no
    --dataset_name HIT_snips_shot_1_out_1 \
    --dataset_path ./data/xval_snips_shot_1_out_1 \
    --deviceId 1
```



