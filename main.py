import torch


from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX", tgt_lang="es_XX")

src_text = " hello world how are you"
tgt_text = "hola mundo como estas"

force_words = ["hola"]

model_inputs = tokenizer(src_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_text, return_tensors="pt").input_ids

model(**model_inputs, labels=labels)  # forward pass
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
article = "hola mundo como estas"
inputs = tokenizer(article, return_tensors="pt")
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

translated_tokens = model.generate(
    **inputs, decoder_start_token_id=tokenizer.lang_code_to_id["es_XX"],
    num_beams=10,
    force_words_ids=force_words_ids
    )
t = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(t)



