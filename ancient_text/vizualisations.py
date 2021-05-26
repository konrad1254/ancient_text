# Implement any vizualisations here

import matplotlib.pyplot as plt

# Wordcloud
from wordcloud import WordCloud

def word_cloud_generator(text_data, name):
    
    str_text_data = " ".join(i for i in text_data)

    wordcloud = WordCloud().generate(str_text_data)

    # Display the generated image:

    # lower max_font_size
    wordcloud = WordCloud(background_color='white', max_font_size=40).generate(str_text_data)
    name = name + ".png"
    wordcloud.to_file(name)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt.show()

