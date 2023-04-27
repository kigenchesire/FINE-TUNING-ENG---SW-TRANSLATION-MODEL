import streamlit as st
import pickle
import torch 
import tokenizer
from transformers import MarianTokenizer,AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel



#pickle_in = open('model.pkl', 'rb') 
#model = pickle.load(pickle_in)
#with open('model.pkl', 'rb') as file:
    #model = torch.load(file, map_location=torch.device('cpu'))
#with open('model.pkl', 'rb') as file:
    #model = pickle.load(model.pkl)
    
model_path = 'model24.pt'  # Update with your own model file path
model22 = torch.load((model_path),map_location=torch.device('cpu'))

#model_name = "Helsinki-NLP/opus-mt-en-swc"
#model = MarianMTModel.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-swc")
model_name = "Helsinki-NLP/opus-mt-en-sw"
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated2= [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    return translated2


# use the translate function in Streamlit
st.title('English To Swahili Translation App')
text_input = st.text_input('Enter the English text to be translated:')
if st.button('Translate'):
    translation = translate(text_input)
    st.write('Translated text:', translation)



