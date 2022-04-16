from torch.utils.data import TensorDataset, DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, PreTrainedTokenizerFast
import json
import pickle
from tqdm import tqdm

x_test = []
y_test = []
with open('spanishval.json') as data_file:
    data = json.load(data_file)
    for i in range(0, len(data)):
        x_test.append(data[i]['translation']['en'])
        y_test.append(data[i]['translation']['es'])


# article_en = "The head of the United Nations says there is no military solution in Syria"
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

for i, x in tqdm(enumerate(x_test)):
    model_inputs = tokenizer(x, return_tensors="pt")

    # # translate from English to Spanish
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
    )

    print(y_test[i], tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="spanishval.json", return_tensors="pt")
# model_inputs = fast_tokenizer

# model_inputs = tokenizer(x_test, return_tensors="pt", padding='longest')
#
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
# )
# print(tokenizer.batch_decode(tokenizer.lang_code_to_id["es_XX"]))

