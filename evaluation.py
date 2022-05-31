import pronouncing
from syltippy import syllabize
from pyverse import Pyverse
from nltk.translate.bleu_score import sentence_bleu

SYLLABLE_WEIGHT = 0.6
BLEU_WEIGHT = 0.4
SYLLABLE_THRESHOLD = 10  # to be removed


def getNumSyllablesEN(sentence):
    lst = sentence.split(' ')
    total_ct = 0
    truncated_word = ""
    last_word = ""
    for word in lst:
        try:
            if truncated_word == "":
                pro_list = pro.phones_for_word(word)
                total_ct += pro.syllable_count(pro_list[0])
                last_word = word
            else:
                attempt_suffix = truncated_word + word
                try:
                    pro_list = pro.phones_for_word(attempt_suffix)
                    syll_count = pro.syllable_count(pro_list[0])
                    total_ct += syll_count
                    last_word = truncated_word
                    truncated_word = ""
                except:
                    attempt_suffix = last_word + attempt_suffix
                    try:
                        pro_list = pro.phones_for_word(attempt_suffix)
                        syll_count = pro.syllable_count(pro_list[0])
                        total_ct -= pro.syllable_count(pro.pronunciations(last_word))
                        total_ct += syll_count
                        last_word = attempt_suffix
                        truncated_word = ""
                    except:
                        truncated_word = truncated_word + word
        except:
            truncated_word = word

    return total_ct


def getNumSyllablesES(word):
    return len(syllabize(word)[0])


def getNumSyllablesESSentence(sentence):
    score = 0
    sentence = sentence.split(" ")
    for word in sentence:
        score += len(syllabize(word)[0])
    return score


def getBleuScore(reference, candidate):
    return sentence_bleu([reference], candidate)

def getRhymeScore(sentence_es, sentence_en):
    es_lines = sentence_es.split('\n')
    en_lines = sentence_en.split('\n')
    

    es_score = 0
    en_score = 0
    rhymeScheme = []
    for i in range(1, len(en_lines)):
        ln = en_lines[i]
<<<<<<< HEAD
        prev = getLastWord(en_lines[i - 1])
        curr = getLastWord(ln)
        rhymeScheme.append(prev in pronouncing.rhymes(curr))

    for i in range(1, len(es_lines)):
        ln = es_lines[i]

        if len(ln) > 1 and len(es_lines[i - 1]) > 1:
=======
        print("EN LINES")
        print(f'{ln}.')
        prev = getLastWord(en_lines[i - 1])
        curr = getLastWord(ln)

        print(prev)
        print(curr)

        rhymeScheme.append(prev in pronouncing.rhymes(curr))

    print(rhymeScheme)
    for i in range(1, len(es_lines)):
        ln = es_lines[i]
        print("ES LINES")
        print(f'{ln}.')

        if len(ln) > 0 and len(es_lines[i - 1]) > 0:
>>>>>>> master
            prev_rhyme = Pyverse(es_lines[i - 1]).consonant_rhyme
            #ores
        

            rhyme = Pyverse(ln).consonant_rhyme
            #oras
            if i < len(rhymeScheme) and not rhymeScheme[i - 1]:
                sim_score = 0
<<<<<<< HEAD
=======
                print("HERE")
>>>>>>> master
            else:
                sim_score = 0
                for j in range(len(prev_rhyme)):
                    sim = 0
                    
                    if j < len(rhyme) and prev_rhyme[j] == rhyme[j]:
                        sim += 1
                    sim_score += sim / len(prev_rhyme)
            es_score += sim_score
        

    return es_score / (len(es_lines) - 1)


def getLastWord(s):
    if ' ' in s:
        return s[s.rindex(' ') + 1:]
    else:
        return ''

<<<<<<< HEAD

=======
def temp(sentence_es, sentence_en):
    print(f'sentence_es = {sentence_es}')
    print(f'sentence_en = {sentence_en}')
>>>>>>> master


def getSyllableScore(sentence_es, sentence_en):
    print(f'sentence_es = {sentence_es}')
    print(f'sentence_en = {sentence_en}')
    english_syll_count = getNumSyllablesEN(sentence_en)
    spanish_syll_count = getNumSyllablesESSentence(sentence_es)
    syllable_score = abs(english_syll_count - spanish_syll_count)
    return syllable_score


def evaluate(sentence_en, sentence_es, sentence_es_actual):
    syllable_score = getSyllableScore(sentence_en, sentence_es)
    bleu_score = getBleuScore(sentence_es, sentence_es_actual)
    return (-pow(syllable_score, 2) * SYLLABLE_WEIGHT + SYLLABLE_THRESHOLD) + bleu_score * BLEU_WEIGHT


if __name__ == "__main__":
    i = 0
    while i < 1:
        # sentence_en =  input("Enter an English sentence: ")
        # sentence_es =  input("Enter an Spanish sentence: ")

        sentence_en =  'it is time to play\n it is time to stay'
        sentence_es =  'es tiempo de jugar\n es tiempo de escoger'
        # print(getNumSyllablesESSentence(sentence_en))
        print(getRhymeScore(sentence_es, sentence_en))

        i = 1





    

