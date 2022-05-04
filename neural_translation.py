from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("TuhinColumbia/spanishpoetrymany")

    model = AutoModelForSeq2SeqLM.from_pretrained("TuhinColumbia/spanishpoetrymany")

    sentence_en = "hello bird"
    model_inputs = tokenizer(sentence_en, return_tensors="pt")
    print("Started generating tokens")
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
    )
    print(len(generated_tokens))
    sentence_es = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print("TRANSLATION", str(sentence_es))
