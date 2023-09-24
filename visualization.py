import matplotlib.pyplot as plt
from config import outputs_path
import pickle
from training import TokenAndPositionEmbedding, TransformerBlock
from wordcloud import WordCloud
from wordcloud import WordCloud
from get_data import PlayerLines

def wordcloud(strings_in_list):
    text = ' '.join(strings_in_list)
    wc = WordCloud(width = 300, height = 300, background_color = "white")
    wc.generate(text)
    plt.axis("off")
    plt.imshow(wc, interpolation = "bilinear")
    plt.savefig(outputs_path + f'/images/wordcloud.png')

# model = tf.keras.models.load_model('./output/model.keras', custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding,'TransformerBlock': TransformerBlock })
#
# model.summary()

# Helper function for plotting model metrics
def plot_metrics(history, string, epoches):
    y = history.history[string]
    y.insert(0,1)
    x = list(map(int,range(epoches+1)))
    plt.plot(y)
    plt.xlabel("Epochs")
    plt.xticks(x)
    plt.ylabel(string)
    plt.savefig(outputs_path + f'/images/{string}.png')

with open(outputs_path + '/history', 'rb') as file:
    history=pickle.load(file)


if __name__ == '__main__':

    # loss for each epoch
    plot_metrics(history, 'loss',3)

    # creating wordcloud
    wordcloud(PlayerLines)
