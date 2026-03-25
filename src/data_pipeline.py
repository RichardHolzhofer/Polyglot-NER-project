import torch
from datasets import DatasetDict, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer

def get_dataset_with_conditions(ds, sources=None, languages=None):
    """
    Filter dataset based on source and language.
    """
    if sources is not None and ds["source"] not in sources:
        return False
    if languages is not None and ds["language"] not in languages:
        return False
    return True

# Master label mapping for harmonizing different datasets
LABEL_MAPPING_DICT = {
    # PERSON
    'full_name': 'PER',
    'first_name': 'PER',
    'last_name': 'PER',
    'middle_name': 'PER',
    'name': 'PER',

    # ORGANIZATION
    'company': 'ORG',
    'organization': 'ORG',
    'hospital_name': 'ORG',

    # LOCATION
    'city': 'LOC',
    'country': 'LOC',
    'state': 'LOC',
    'street_address': 'LOC',
    'postal_code': 'LOC',
    'county': 'LOC',
}

def ner_spans_to_bio_tags_batched(batch, ner_col, mapping_dict=LABEL_MAPPING_DICT):
    """
    Converts word-span annotations to standardized labels.
    Only maps labels present in the mapping_dict; others remain as 'O'.
    """
    standardized_ner_list = []
    
    for list_of_spans in batch[ner_col]:
        standardized_list = []
        for span in list_of_spans:
            if len(span) >= 3:
                start, end, label = span[0], span[1], span[2]
                if label in mapping_dict:
                    new_label = mapping_dict[label]
                    standardized_list.append([start, end, new_label])
                    
        standardized_ner_list.append(standardized_list)
        
    batch[ner_col] = standardized_ner_list
    return batch

def convert_spans_to_ids_batched(batch, token_col, ner_col, master_label2id):
    """
    Converts standardized spans into BIO tag integer IDs aligned with word tokens.
    """
    all_bio_tags = []
    
    for tokens, list_of_spans in zip(batch[token_col], batch[ner_col]):
        # Initializing 'O' tags (ID 0)
        num_tokens = len(tokens)
        bio_tags = [0] * num_tokens
        
        # Mapping tags
        for span in list_of_spans:
            # span = [start_idx, end_idx, 'label']
            start, end, label = span[0], span[1], span[2]
            
            # Inserting tags with proper B- and I- prefixes
            if start < num_tokens:
                bio_tags[start] = master_label2id[f"B-{label}"]
                for i in range(start + 1, min(end + 1, num_tokens)):
                    bio_tags[i] = master_label2id[f"I-{label}"]
                    
        all_bio_tags.append(bio_tags)
        
    batch['ner'] = all_bio_tags
    return batch

def map_ger_tags_to_master(batch, ner_col, ger_id2label, master_label2id):
    """
    Maps German dataset BIO tag IDs to the Master Hungarian-based ID schema.
    """
    all_new_tags = []
    for sentence_tags in batch[ner_col]:
        new_sentence_tags = [master_label2id[ger_id2label[tag_id]] for tag_id in sentence_tags]
        all_new_tags.append(new_sentence_tags)
    
    batch["ner"] = all_new_tags
    return batch

def align_labels_to_tokenized_text(batch, tokenizer, token_col, ner_col):
    """
    Aligns word-level BIO tags with sub-word tokenization.
    Uses the 'first-subtoken only' approach, assigning -100 to follow-up sub-tokens.
    """
    tokenized_inputs = tokenizer(
        batch[token_col], 
        truncation=True, 
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(batch[ner_col]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like <s> or </s>
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # This is the FIRST sub-token of a new word
                label_ids.append(label[word_idx])
            else:
                # This is a follow-up sub-token of the same word
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_master_schema(features):
    """
    Generates id2label and label2id mappings from a dataset split's features.
    """
    label_names = features['ner'].feature.names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    return label_names, id2label, label2id
