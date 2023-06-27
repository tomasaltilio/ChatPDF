import os
import re
import shutil
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz
import numpy as np
import openai
import tensorflow_hub as hub
import tensorflow_text
from fastapi import UploadFile
from sklearn.neighbors import NearestNeighbors
import requests


recommender = None

pdf_source = 'basket_rules.pdf'

path_enc = 'https://tfhub.dev/google/'
multi = 'universal-sentence-encoder-multilingual/3'
eng = 'universal-sentence-encoder/4'
lang = 'eng'

# Define encoder
if lang != 'eng':
    encoder = path_enc + multi
else:
    encoder = path_enc + eng

falcon = \
    "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self):
        self.use = hub.load(encoder)
        self.fitted = False

    def fit(self, data, batch=500, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    try:
        completions = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7,
        )
        message = completions.choices[0].text
    except Exception as e:
        message = f'API Error: {str(e)}'
    return message 


def generate_answer(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, "text-davinci-003")
    return answer


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "[ERROR]: Please pass your OPENAI_API_KEY."
        )
    return key


def generate_answer_falcon(question):
    API_URL = falcon
    key = os.environ.get('HF_API_KEY')
    headers = {"Authorization": f"Bearer {key}"}
    
    prompt = """
    Answer the question as truthfully as possible using the provided text, 
    and if the answer is not contained within the text below, say "I don't know"
    """
    
    topn_chunks = recommender(question)
    prompt += '\nContext:\n'
    for c in topn_chunks:
        prompt += c + '\n'

    prompt += question

    prompt = {'inputs': prompt}

    response = requests.post(API_URL, headers=headers, json=prompt)
    
    content = response.json()[0]['generated_text']
    
    for _ in range(10):
        prompt = {'inputs': content}
        response = requests.post(API_URL, headers=headers, json=prompt)
        if response.status_code != 200:
            break
        
        if content == response.json()[0]['generated_text']:
            break
        
        content = response.json()[0]['generated_text']

    answer = content.split(question)[-1].strip('\n')

    return answer
	


def ask_file(file: UploadFile, question: str) -> str:
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    load_recommender(str(tmp_path))
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)


if __name__ == '__main__':
    
    load_recommender(pdf_source)
    
    while True:
        try:
            user_input = input("You: ")
            # openAI_key = load_openai_key()
            # output = generate_answer(user_input, openAI_key)
            output = generate_answer_falcon(user_input)
            print(f"Chat: {output}\n")
        except KeyboardInterrupt:
            print("\n\nBye!")
            break