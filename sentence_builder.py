sentence = ""

def add_letter(letter):
    global sentence
    if isinstance(letter, str) and len(letter) == 1:
        sentence += letter

def add_space():
    global sentence
    sentence += " "

def backspace():
    global sentence
    if len(sentence) > 0:
        sentence = sentence[:-1]

def clear_sentence():
    global sentence
    sentence = ""

def get_sentence():
    return sentence
