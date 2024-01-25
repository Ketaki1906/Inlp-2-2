import re
import sys
from collections import defaultdict
import random

with open(sys.argv[2], 'r', encoding='utf-8') as file:
    text_JA = file.read()
# text_JA=" "
# text="Hello Mr. and Mrs. Jones, meet Dr. David. He is an expert surgeon."

# print(text_JA)
lm_type= sys.argv[1]

#(a)Sentence Tokeniser
def sentence_tokenizer(text):
    sentences = re.split(r'(?<!Mr\.)(?<!Mrs\.)(?<!Dr\.)(?<![A-Z].)(?<!\[A-Za-z]\.\[A-Za-z]\.)(?<=\.|\?)\s',text)   # Handling cases when tokens like 'Mr.' or 'S.' or 'gmail.com' or 'U.S.'
    sentences = [re.sub(r'["\n"]',' ', sentence) for sentence in sentences]
    return sentences

#(b)Word Tokeniser
def word_tokeniser(text):
    words=[]
    sentences= sentence_tokenizer(text)
    for sentence in sentences:
        words_in_sentence=re.split(r'(?<=[A-Za-z0-9"])\s',sentence)     # Numbers and words made of alphabets immediately followed by punctuation marks are treated as different words
        words.append(words_in_sentence)
    return words

# 2. N-gram Model

def remove_punc(sentences):
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

def n_gram(sentences, n):
    n_gram ={}

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

# n_gram_model=n_gram(result_JA,n)

all_sentences=sentence_tokenizer(text_JA)

held_out_percentage = 0.10                       # 10% of the total sentences for held-out data
held_out_count = int(len(all_sentences) * held_out_percentage)
held_out_data = random.sample(all_sentences, held_out_count)

training_data = [sentence for sentence in all_sentences if sentence not in held_out_data]     # Remove held-out sentences from the original list of sentences

test_data_count = 1000                            # Select 1000 sentences randomly as test data
test_data = random.sample(training_data, test_data_count)

train_data = [sentence for sentence in training_data if sentence not in test_data]

# Print or use the data as needed
# print("Held-out data:\n", held_out_data)
# print("Test data:", test_data)
# print("Final training data:", train_data)

# Finding appropiate weights for lambda_1,lambda_2,lambda_3 using EM method using held-out data

iterations_to_be_carried=10

def calculate_expected_counts(data_with_punc,l_1,l_2,l_3):
    expected_counts={}
    # expected_counts=defaultdict(int)
    data=remove_punc(data_with_punc)
    # print(data)
    three_gram=n_gram(data,3)
    two_gram=n_gram(data,2)
    one_gram=n_gram(data,1)
    # print(len(three_gram))
    for trigram,trigram_frequency in three_gram.items():
        tri_cond = trigram[0:2]
        bigram = trigram[1:3]                         
        # print(bigram)
        bi_cond=trigram[1:2]
       
#         print(three_gram[trigram])
#         print(bigram)
#         print(f"{bi_cond}{one_gram[bi_cond]}")
        i_prob=l_1*three_gram[trigram]/two_gram[tri_cond]+(l_2*two_gram[bigram]/one_gram[bi_cond])+(l_3)
        expected_counts[trigram]=i_prob*three_gram[trigram]
    # print(expected_counts)
        # break
    return expected_counts

def update_weights(expected_counts,data_with_punc):
    expected_count=0
    # print(expected_counts)
    # print(data_with_punc)
    data=remove_punc(data_with_punc)
    three_gram=n_gram(data,3)
    two_gram=n_gram(data,2)
    one_gram=n_gram(data,1)
    for trigram,e_count in three_gram.items():
        # print(trigram[0:3])
        # print(e_count)
        # print(expected_counts[trigram])
        # print(three_gram[trigram])
        expected_count=expected_count+expected_counts[trigram[0:3]]*three_gram[trigram]
    total_actual_count=0
    for trigram,a_count in three_gram.items():
        total_actual_count=total_actual_count+three_gram[trigram]

    l_1= expected_count/total_actual_count

    expected_count=0
    for trigram,e_count in three_gram.items():
        bigram = trigram[1:3]                 
        expected_count=expected_count+expected_counts[trigram]*two_gram[bigram]

    l_2=expected_count/total_actual_count

    for trigram,e_count in three_gram.items():
        unigram=trigram[2:3]
        expected_count=expected_count+expected_counts[trigram]*one_gram[unigram]

    l_3=expected_count/total_actual_count

    total_weight=l_1+l_2+l_3
    l_1=l_1/total_weight
    l_2=l_2/total_weight
    l_3=l_3/total_weight

    return l_1,l_2,l_3

def calculating_final_weights(held_out_data,iterations_to_be_carried):
    l_1,l_2,l_3=0.33,0.33,0.33
    for iteration in range(iterations_to_be_carried):
        expected_counts=calculate_expected_counts(held_out_data,l_1,l_2,l_3)
        # print(expected_counts)
        l_1,l_2,l_3=update_weights(expected_counts,held_out_data)
    return l_1,l_2,l_3

l_1,l_2,l_3=calculating_final_weights(held_out_data,iterations_to_be_carried)

# print(l_1)
# print(l_2)
# print(l_3)

def sentence_prob(sentence_with_punc,train_data_with_punc,l_1,l_2,l_3):
    sentence=remove_punc(sentence_with_punc)
    # print(sentence)
    three_gram_input_sent=n_gram(sentence,3)
    data=remove_punc(train_data_with_punc)
    # print(data)
    three_gram=n_gram(data,3)
    
    two_gram=n_gram(data,2)
    # print(two_gram)
    one_gram=n_gram(data,1)
    total_unigram_count=len(one_gram)
    # print(one_gram)
    total_prob=1
    # print(three_gram_input_sent)
    for trigram,trigram_frequency in three_gram_input_sent.items():
        tri_cond = trigram[0:2]
        bigram = trigram[1:3]                                
        bi_cond=trigram[1:2]
        unigram=trigram[2:3]
        uni_cond=trigram[2:2]
        if trigram in three_gram:
            # print("trigram")
            # print(three_gram[trigram])
            # trigram=trigram
            # tri_cond=tri_cond
            i_prob=l_1*three_gram[trigram]/two_gram[tri_cond]+(l_2*two_gram[bigram]/one_gram[bi_cond])+(l_3)*(one_gram[unigram]/total_unigram_count)
        elif bigram in two_gram:
            # print("bigram")
            # print(two_gram[bigram])
            i_prob=l_1*two_gram[bigram]/one_gram[bi_cond]+(l_2*two_gram[bigram]/one_gram[bi_cond])+(l_3)*(one_gram[unigram]/total_unigram_count)
        elif unigram in one_gram:
            # print("unigram")
            trigram=unigram
            tri_cond=uni_cond
            bigram=unigram
            bi_cond=uni_cond
            i_prob=(l_1+l_2+l_3)*one_gram[unigram]/total_unigram_count
        else:
            # print("constant")          
            i_prob=(l_1+l_2+l_3)*0.000001
        total_prob=total_prob*i_prob
        # print(total_prob)
    return total_prob


for sentence in test_data:
    print(sentence)
    break
str=input("Enter Sentence: ")
sent=[]
sent.append(str)
# prob=sentence_prob(sent,train_data,l_1,l_2,l_3)
# print(prob)

def finding_count(ngram,c):
    n_list=[]
    for key,value in ngram.items():
        if(value==c):
            n_list.append(key)
    print(len(n_list))
    return n_list

def adjusting_count(c,ngram):
    c_new=(c+1)*len(finding_count(ngram,c+1))/len(finding_count(ngram,c))
    print(c_new)
    return c_new

def count_trigram_threegram(three_gram,trigram):
    for key,value in three_gram.items():
        if(key==trigram):
            return value
    k=0
    return float(k)  

def sentence_prob_gt(train_data,sentence_with_punc):
    sentence=remove_punc(sentence_with_punc)
    three_gram_sent=n_gram(sentence,3)
    three_gram=n_gram(train_data,3)
    two_gram=n_gram(train_data,2)
    one_gram=n_gram(train_data,1)
    total_prob=1
    for trigram,value in three_gram_sent.items():
        print(trigram)
        c=count_trigram_threegram(three_gram,trigram)
        if(c==0):
            total_prob=total_prob*adjusting_count(1,three_gram)/len(three_gram)
            print(adjusting_count(1,three_gram)/len(three_gram))
        else:
            c_new=adjusting_count(c,three_gram)
            prob=c_new/len(three_gram)
            total_prob=total_prob*prob
    return total_prob

total_prob=sentence_prob_gt(train_data,sent)
print(total_prob)
