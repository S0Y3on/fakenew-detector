from django.shortcuts import render
from .forms import CoverForm
from .models import Main
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
from .make_models import *

make_models()

def to_one_hot(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results



loaded_model = load_model('datatext_binary_model.h5')

print("model loaded:", loaded_model)

with open('datatext_binary_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


def main(request):

    if request.method == 'POST':
        form = CoverForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data['input_text'])
            input_text = [form.cleaned_data['input_text']]

            data = loaded_tokenizer.texts_to_sequences(input_text)
            data = pad_sequences(data, maxlen=200)
            x_test = to_one_hot(data, dimension=10000)

            prediction = loaded_model.predict(x_test)
            print("Result:", prediction)

            percent = int(prediction[0][0]*100)
            r = 'Unable to judge'
            if percent<40:
                r = 'False'
            elif percent>=70:
                r = 'True'

            Main.result = round(prediction[0][0]*100,2)

            url = "https://search.naver.com/search.naver?query=%s&where=news&ie=utf8&sm=nws_hty" % form.cleaned_data[
                'input_text']

            result_data = {'form': form, 'result': Main.result, 'input_text': form.cleaned_data['input_text'], 'url':url, 'r':r}

            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            ranking_text = soup.find_all('a', class_="news_tit")
            count = 0
            for a in ranking_text:
                href = a.attrs['href']
                text = a.attrs['title']
                result_data['news_text' + str(count)] = text
                result_data['news_href' + str(count)] = href
                count += 1
                print(text, ":", href)
            return render(request, 'factsignal/result.html', result_data)
            # return render(request, 'factsignal/result.html', {'form': form, 'result': Main.result,'input_text' : input_text })
    else:
        form = CoverForm()
        return render(request, 'factsignal/index.html', {'form': form})
