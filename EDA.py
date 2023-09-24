import matplotlib.pyplot as plt
from wordcloud import WordCloud
from get_data import PlayerLines
import nltk
from nltk.corpus import stopwords
import string

sw_nltk = stopwords.words('english')
# dataset as a long text
conc_text = ' '.join(PlayerLines)
# removing  punctuation marks
conc_text_no_punct = conc_text.translate(str.maketrans('', '', string.punctuation))
# making all data lowercase
conc_text_low = [word.lower() for word in conc_text_no_punct.split()]
unique_words = set(conc_text_low)
print(unique_words)

length = len(unique_words)
print(length)
#
unique_words_without_stopwords = [word for word in unique_words if word not in sw_nltk]
#
print("Old length: ", length)
print("New length: ", len(unique_words_without_stopwords))