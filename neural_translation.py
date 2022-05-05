from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
if __name__ == "__main__":
    tokenizer = MBart50TokenizerFast.from_pretrained("TuhinColumbia/spanishpoetrymany")
    model = MBartForConditionalGeneration.from_pretrained("TuhinColumbia/spanishpoetrymany")
    tokenizer.src_lang = "es_XX"
    sentence_en = "del pan que nos enseñas a ganárnoslo"
    model_inputs = tokenizer(sentence_en, return_tensors="pt")
    print("Started generating tokens")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    print(len(generated_tokens))
    sentence_es = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print("TRANSLATION", str(sentence_es))
