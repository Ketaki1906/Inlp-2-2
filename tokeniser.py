import re
import sys
from collections import defaultdict

with open(sys.argv[1], 'r', encoding='utf-8') as file:
    text_JA = file.read()
# text_JA=" "
# text="Hello Mr. and Mrs. Jones, meet Dr. David. He is an expert surgeon."

n = sys.argv[2]
n = int(n)

#(a)Sentence Tokeniser
def sentence_tokenizer(text):
    sentences = re.split(r'(?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<![A-Z].)(?<!\[A-Za-z]\.\[A-Za-z]\.)(?<=\.|\?)\s',text)   # Handling cases when tokens like 'Mr.' or 'S.' or 'gmail.com' or 'U.S.'
    sentences = [re.sub(r'["\n"]', ' ', sentence) for sentence in sentences]
    return sentences


#(b)Word Tokeniser
def word_tokeniser(text):
    words=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words_in_sentence=re.split(r'(?<=[A-Za-z0-9,:;.!)({}"])\s',sentence)     # Numbers and words made of alphabets immediately followed by punctuation marks are treated as different words
        words.append(words_in_sentence)
    return words


# sentences_JA= sentence_tokenizer(text_JA)
# print(sentences_JA)
print("\n")

words_JA=word_tokeniser(text_JA)
# print(words_JA)
# print("\n")

#(c)Numbers

def number(text):
    numbers=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        numbers_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern = r'\b\d+,?.?\d*\b'                                 # Finding integer,hyphenated,comma-seperated as well as decimel numbers
                num = re.findall(pattern,word)
                num = [n for n in num if n]
            
                numbers_in_sentence.extend(num)
        numbers.append(numbers_in_sentence)

    return numbers

# numbers_JA=number(text_JA)
# print(numbers_JA)                                                          # Printing the numbers identified in each sentence as a list
# print("\n")

#(d)Mail Ids

def mail_id(text):
    mail_id=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        mail_id_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\b\w+@\w+.\w+\b$'
                emails=re.findall(pattern,word)
                emails = [n for n in emails if n]
                if emails:
                    mail_id_in_sentence.extend(emails)
        mail_id.append(mail_id_in_sentence)
    return mail_id

# email_JA=mail_id(text_JA)
# print(email_JA)
# print("\n")

#Punctuation Marks

def punc(text):
    punct=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        punct_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'[^\w\s]'
                puncts=re.findall(pattern,word)
                puncts = [n for n in puncts if n]
                if puncts:
                    punct_in_sentence.extend(puncts)
        punct.append(punct_in_sentence)
    return punct

punct_JA=punc(text_JA)
print(punct_JA)
# print("\n")

#URLs

def url(text):
    urls=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        url_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\bhttp?://\S+|www.\S+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    url_in_sentence.extend(links)
        urls.append(url_in_sentence)
    return urls

# url_JA=url(text_JA)
# print(url_JA)
# print("\n")

#HashTags

def hash_tag(text):
    hash=[]
    # count=0
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        hash_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\b#\w+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    # print(links)
                    # count=count+len(links)
                    hash_in_sentence.extend(links)
        hash.append(hash_in_sentence)
    return hash

# hash_JA=hash_tag(text_JA)
# print(hash_JA)
# print("\n")

#HashTags

def mention(text):
    mentions=[]
    # count=0
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words=word_tokeniser(sentence)
        mention_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'^\b@\w+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    # print(links)
                    # count=count+len(links)
                    mention_in_sentence.extend(links)
        mentions.append(mention_in_sentence)
    return mentions

# mention_JA=mention(text_JA)
# print(mention_JA)
# print("\n")

# Replacement with Placeholder 

# (a) Replace url
def replace_url(text):
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        count=0
        words=word_tokeniser(sentence)
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\bhttp?://\S+|www.\S+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    count=1
                    links=links[0]
                    sentence=sentence.replace(links,"<URL>")
        if count==1:
            count=0
            print(sentence)

replace_url(text_JA)
print("\n")

# (b) Replace Hashtags

def replace_hashtags(text):
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        count=0
        words=word_tokeniser(sentence)
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\b#\w+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    count=1
                    links=links[0]
                    sentence=sentence.replace(links,"<HASHTAG>")
        if count==1:
            count=0
            print(sentence)

replace_hashtags(text_JA)
print("\n")

#(c) Replace Mentions

def replace_mentions(text):
    
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        count=0
        words=word_tokeniser(sentence)
        mention_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'^\b@\w+'
                links=re.findall(pattern,word)
                links = [n for n in links if n]
                if links:
                    count=1
                    links=links[0]
                    sentence=sentence.replace(links,"<MENTION>")
        if count==1:
            count=0
            print(sentence) 

replace_mentions(text_JA)
print("\n")

# (d) Replace Numbers

def replace_number(text):
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        count=0
        words=word_tokeniser(sentence)
        numbers_in_sentence=[]
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern = r'\b\d+,?.?\d*\b'                                 # Finding integer,hyphenated,comma-seperated as well as decimel numbers
                num = re.findall(pattern,word)
                num = [n for n in num if n]
                if num:
                    count=1
                    num=num[0]
                    sentence=sentence.replace(num,"<NUM>")
        if count==1:
            count=0
            print(sentence) 

replace_number(text_JA)
print("\n")

# (e) Replace Mail_Ids

def replace_mail_id(text):
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        count=0
        words=word_tokeniser(sentence)
        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                pattern=r'\b\w+@\w+.\w+\b$'
                emails=re.findall(pattern,word)
                emails = [n for n in emails if n]
                if emails:
                    count=1
                    emails=emails[0]
                    sentence=sentence.replace(emails,"<MAILID>")
        if count==1:
            count=0
            print(sentence) 

replace_mail_id(text_JA)
print("\n")

# 2. N-gram Model

def remove_punc(text):
    sentences = sentence_tokenizer(text)
    text_without_punct = []

    for sentence in sentences:
        words = word_tokeniser(sentence)
        clean_words = []

        for word_list_in_sentence in words:
            for word in word_list_in_sentence:
                # Remove punctuation marks
                clean_word = re.sub(r'[^\w\s]', '', word)
                clean_words.append(clean_word)

        clean_sentence = ' '.join(clean_words)
        text_without_punct.append(clean_sentence)

    return text_without_punct

from collections import defaultdict

def n_gram(sentences, n):
    n_gram = {}

    for sentence in sentences:
        words = word_tokeniser(sentence)
        for words_in_sent in words:
            number_of_words = len(words_in_sent)

            if n <= number_of_words:
                start = 0
                while start < (number_of_words - n + 1):
                    prefix = tuple(words_in_sent[start : start + n])
                    start = start + 1

                    if prefix in n_gram:
                        n_gram[prefix] = n_gram[prefix] + 1
                    else:
                        n_gram[prefix] = 1

    return n_gram


result_JA = remove_punc(text_JA)
# print(result_JA)
n_gram_model=n_gram(result_JA,n)
# print(n_gram_model)
