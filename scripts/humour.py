from transformers import BertTokenizer
import keras
import nltk
import numpy as np
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import tensorflow as tf

MODEL_TYPE = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

#options_model = tf.saved_model.LoadOptions(
#    experimental_io_device=None
#)


#model = keras.models.load_model("./models/colbert-trained/", options=options_model)
print('Loading humour model')
model = keras.models.load_model("./models/colbert-trained/")
print(model.summary())

MAX_SENTENCE_LENGTH = 20
MAX_SENTENCES = 5
MAX_LENGTH = 100


def set_global_vars(max_length, max_sentences):
    MAX_SENTENCES = max_sentences
    MAX_LENGTH = max_length


def return_id(str1, str2, truncation_strategy, length):

    inputs = tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, column):
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])
    
    # for _, row in tqdm(df[column].iterrows()):
    for _, row in tqdm(df[column].items()):
        i = 0
        
        # sent
        sentences = sent_tokenize(row)
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        # full row
        ids_q, masks_q, segments_q = return_id(row, None, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    print(model_input[0].shape)
    return model_input


def predict(inputs):
    return model.predict(inputs)
