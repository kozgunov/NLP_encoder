from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import nltk

# last version
model_path = f'saved_models/2023-11-02_12-55-16/model'
tokenizer_path = f'saved_models/2023-11-02_12-55-16/tokenizer'

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))  # russian stop words



def clean_text(text):  # removing the links, transforming to lower register, removing the symbols
    #text = re.sub(r'http\S+', ' ', text)
    #text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
    #text = re.sub(r'\n', ' ', text).strip() #- only if we wanna remove whole symbols
    return text.lower()


def remove_stop_words(text):  # removing some russian stop words
    cleaned_text_list = []
    for text in text:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_text_list.append(' '.join(filtered_words))
    return cleaned_text_list


def interactive_sentiment_analysis():  # load trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(f'C:/Users/user/pythonProject/{model_path}')
    tokenizer = BertTokenizer.from_pretrained(f'C:/Users/user/pythonProject/{tokenizer_path}')

    while True:  # interactive loop

        user_input = input("input: ")  # end
        if user_input.lower() == 'exit':
            break

        # transform input text
        cleaned_texts = clean_text(user_input)

        encoding = tokenizer(cleaned_texts, truncation=True, padding=True, max_length=1023, return_tensors='pt')  # 1023 is max length of sentence

        with torch.no_grad():
            outputs = model(**encoding)
        predicts = torch.argmax(outputs.logits, dim=1).item()  # model's prediction

        sentiment_label = {0: 'positive', 1: 'neutral', 2: 'negative'}
        sentiment = sentiment_label[predicts]  # convert the reply
        print(f"output: {sentiment}")


# running
if __name__ == "__main__":
    interactive_sentiment_analysis()

# model, which get text as input and response the mood of it (0,1,2)
'''
def get_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=1023, return_tensors='pt') # 1024+ were cut off
    with torch.no_grad(): # turn the gradient response off
        outputs = model(**encoding)
        preds = torch.argmax(outputs.logits, dim=1).item() # use softmax for output
    sentiment_dict = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return sentiment_dict[preds]
'''


# didn't use context because of time...
