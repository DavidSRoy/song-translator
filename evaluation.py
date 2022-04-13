import pronouncing as pro

print('hello')




def getNumSyllables(word):
    pro_list = pro.phones_for_word(word)
    return pro.syllable_count(pro_list[0])

while True:
    word = input("Enter a word: ")
    num = getNumSyllables(word)
    print(f'Number of syllables: {num}')
    