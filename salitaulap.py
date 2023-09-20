from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import stws
from nltk.corpus import stopwords

# Reads 'Youtube04-Eminem.csv' file


def visualize(dict, x, y):
    df = pd.DataFrame(dict)
    df.plot(x=x, y=y, kind="bar", ylim=(.6, 1))
    plt.show()


def generateWordCloud(df):
    data = df.to_dict()['Total']
    stop_words = stopwords.words("english")
    custom = ["also", "would", "will"]
    stop_words.extend(custom)

    # iterate through the csv file
    # for txt in df[data]:
    #
    #     # typecaste each val to string
    #     txt = str(txt)
    #
    #     # split the value
    #     tokens = stws.getTokens(txt)
    #     #
    #     # # Converts each token into lowercase
    #     # for i in range(len(tokens)):
    #     #     tokens[i] = tokens[i].lower()
    #
    #     features.extend(tokens)

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate_from_frequencies(frequencies=data)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


# df = pd.read_csv(r"ALL_DATA(oversampled).csv", encoding="latin-1")
# generateWordCloud(df, "Backgrounds")
