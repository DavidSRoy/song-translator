from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from evaluation import getSyllableScore, getBleuScore
NUM_TO_TRANSLATE = 30
NUM_BEAMS = 4
INPUT_LANG_CODE = "es_XX"
OUTPUT_LANG_CODE = "en_XX"
LOGS_ON = True

if __name__ == "__main__":
    tokenizer = MBart50TokenizerFast.from_pretrained("TuhinColumbia/spanishpoetrymany")
    model = MBartForConditionalGeneration.from_pretrained("TuhinColumbia/spanishpoetrymany")
    tokenizer.src_lang = "es_XX"
    sentence_en_gold = "You, myself, dry like a defeated wind \n which only for a moment could hold in its arms the leaf \n it wrenched from the trees, \n how is it possible that nothing can move you now,\n that no rain can crush you, no sun give back your weariness? \n To be a purposeless transparency"
    sentence_es = "A pesar del perífrasis absurdo"
    model_inputs = tokenizer(sentence_es, return_tensors="pt")
    print("Started generating tokens")
    generated_tokens = model.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[OUTPUT_LANG_CODE],
            num_beams=NUM_BEAMS,
            num_beam_groups=NUM_BEAMS//2,
            num_return_sequences=NUM_BEAMS,
            diversity_penalty=2.0
    )
    print(len(generated_tokens))
    sentence_en = tokenizer.batch_decode(generated_tokens)
    sentence_en_skip = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(sentence_en)
    print("---------------")
    print(sentence_en_skip)
    '''
    best_candidate_score = float('inf')
    best_candidate = None
    for s in sentence_en:
        output_joined = s.strip()
        print(output_joined)
        score = getSyllableScore(sentence_es, output_joined)
        if score < best_candidate_score:
            best_candidate_score = score
            best_candidate = s
    print("BLEU SCORE: "+str(getBleuScore(sentence_en_gold, best_candidate)))
    '''