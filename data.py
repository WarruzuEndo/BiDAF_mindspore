import os
import shutil
import tempfile
import zipfile
import json
from typing import IO
from pathlib import Path
import nltk
import requests
import numpy as np
from tqdm import tqdm

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter


cache_dir = Path.home() / '.mindspore_examples'


def http_get(url: str, temp_file: IO):
    """
    Use requests to download dataset
    """
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download(file_name: str, url: str):
    """
    Download dataset
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, file_name)
    cache_exist = os.path.exists(cache_path)
    if not cache_exist:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    return cache_path


def load_glove(embed_path):
    """
    load glove
    """
    glove_100d_path = os.path.join(cache_dir, 'glove.6B.100d.txt')
    if not os.path.exists(glove_100d_path):
        glove_zip = zipfile.ZipFile(embed_path)
        glove_zip.extractall(cache_dir)

    glove_embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            glove_embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
    glove_embeddings.append(np.random.rand(100))
    glove_embeddings.append(np.zeros((100,), np.float32))

    glove_vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    glove_embeddings = np.array(glove_embeddings).astype(np.float32)
    return glove_vocab, glove_embeddings


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))
            return dump

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Iterator():
    def __init__(self, dataset, word_vocab, char_vocab,
                 max_c_word_len, max_q_word_len, max_char_len):
        self.dataset = dataset
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_c_word_len = max_c_word_len
        self.max_q_word_len = max_q_word_len
        self.max_char_len = max_char_len

        self._load()

    def _load(self):
        def pad_word(word_list, word_vocab, max_word_len):
            if len(word_list) < max_word_len:
                for _ in range(len(word_list), max_word_len):
                    word_list.append("<pad>")
            return word_vocab.tokens_to_ids(word_list)

        def pad_char(char_list, char_vocab, max_word_len, max_char_len):
            char_pad_list = []
            for char in char_list:
                if len(char) < max_char_len:
                    for _ in range(len(char), max_char_len):
                        char.append("<pad>")
                    char_pad_list.append(char_vocab.tokens_to_ids(char))

            if len(char_pad_list) < max_word_len:
                for _ in range(len(char_pad_list), max_word_len):
                    char_pad_list.append(char_vocab.tokens_to_ids(["<pad>"] * max_char_len))
            return char_pad_list

        self.ids = []
        self.c_lens = []
        self.q_lens = []
        self.s_idx = []
        self.e_idx = []
        c_char_list = []
        q_char_list = []
        c_word_list = []
        q_word_list = []

        for data in self.dataset:
            ids = data["id"]
            s_idx = data["s_idx"]
            e_idx = data["e_idx"]
            context = data["context"]
            question = data["question"]

            c_word = word_tokenize(context)
            q_word = word_tokenize(question)
            c_lens = len(c_word)
            q_lens = len(q_word)
            c_char = []
            q_char = []
            for word in c_word:
                c_char.append(list(word))
            for word in q_word:
                q_char.append(list(word))

            c_word = pad_word(c_word, self.word_vocab, self.max_c_word_len)
            q_word = pad_word(q_word, self.word_vocab, self.max_q_word_len)
            c_char = pad_char(c_char, self.char_vocab, self.max_c_word_len, self.max_char_len)
            q_char = pad_char(c_char, self.char_vocab, self.max_q_word_len, self.max_char_len)

            c_char = np.array(c_char, dtype=np.int32)
            q_char = np.array(q_char, dtype=np.int32)
            c_word = np.array(c_word, dtype=np.int32)
            q_word = np.array(q_word, dtype=np.int32)

            c_char_list.append(c_char)
            q_char_list.append(q_char)
            c_word_list.append(c_word)
            q_word_list.append(q_word)
            self.ids.append(ids)
            self.c_lens.append(c_lens)
            self.q_lens.append(q_lens)
            self.s_idx.append(s_idx)
            self.e_idx.append(e_idx)

        self.c_char = np.stack(c_char_list, axis=0)
        self.q_char = np.stack(q_char_list, axis=0)
        self.c_word = np.stack(c_word_list, axis=0)
        self.q_word = np.stack(q_word_list, axis=0)
        print(self.c_char.shape)

    def __getitem__(self, index):
        return self.c_char[index], self.q_char[index], self.c_word[index], self.q_word[index],\
               self.c_lens[index], self.q_lens[index], self.s_idx[index], self.e_idx[index], self.ids[index]

    def __len__(self):
        return len(self.c_char)


def download_dataset():
    print("preprocessing data files...")
    train_path = '.data/train-v1.1.json'
    valid_path = '.data/dev-v1.1.json'

    return SQuAD(train_path), SQuAD(valid_path)


def build_char_vocab(train_dataset, valid_dataset):
    print("building vocab...")
    chars = []
    for item in train_dataset:
        context_word = word_tokenize(item["context"])
        question_word = word_tokenize(item["question"])
        for word in context_word:
            char = list(word)
            for c in char:
                chars.append(c)
        for word in question_word:
            char = list(word)
            for c in char:
                chars.append(c)

    for item in valid_dataset:
        context_word = word_tokenize(item["context"])
        question_word = word_tokenize(item["question"])
        for word in context_word:
            char = list(word)
            for c in char:
                chars.append(c)
        for word in question_word:
            char = list(word)
            for c in char:
                chars.append(c)

    chars = list({}.fromkeys(chars).keys())

    return ds.text.Vocab.from_list(chars, special_tokens=["<unk>", "<pad>"])


if __name__ == "__main__":
    # load datasets
    glove_path = download('glove.6B.zip',
                          'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/glove.6B.zip')
    word_vocab, embeddings = load_glove(glove_path)

    train_dataset, valid_dataset = download_dataset()
    char_vocab = build_char_vocab(train_dataset, valid_dataset)
    char_vocab_size = len(char_vocab.vocab())

    print("building iterators...")
    train_iterator = Iterator(train_dataset, word_vocab, char_vocab,
                            max_c_word_len=768, max_q_word_len=64, max_char_len=48)
    valid_iterator = Iterator(valid_dataset, word_vocab, char_vocab,
                            max_c_word_len=768, max_q_word_len=64, max_char_len=48)

    nlp_schema = {
        "c_char": {"type": "int32", "shape": [768, 48]},
        "q_char": {"type": "int32", "shape": [64, 48]},
        "c_word": {"type": "int32", "shape": [768]},
        "q_word": {"type": "int32", "shape": [64]},
        "c_lens": {"type": "int32"},
        "q_lens": {"type": "int32"},
        "s_idx": {"type": "int32"},
        "e_idx": {"type": "int32"},
        "ids": {"type": "string"}
    }

    train_datalist = []
    valid_datalist = []
    for item in train_iterator:
        sample = {
        "c_char": item[0],
        "q_char": item[1],
        "c_word": item[2],
        "q_word": item[3],
        "c_lens": item[4],
        "q_lens": item[5],
        "s_idx": item[6],
        "e_idx": item[7],
        "ids": item[8]
        }
        train_datalist.append(sample)

    for item in valid_iterator:
        sample = {
        "c_char": item[0],
        "q_char": item[1],
        "c_word": item[2],
        "q_word": item[3],
        "c_lens": item[4],
        "q_lens": item[5],
        "s_idx": item[6],
        "e_idx": item[7],
        "ids": item[8]
        }
        valid_datalist.append(sample)

    train_writer = FileWriter(file_name=".data/train.mindrecord",
                            shard_num=1, overwrite=True)
    train_writer.add_schema(nlp_schema, "preprocessed squad train dataset")
    train_writer.write_raw_data(train_datalist)
    train_writer.commit()
    print("squad train data write success")

    valid_writer = FileWriter(file_name=".data/dev.mindrecord",
                            shard_num=1, overwrite=True)
    valid_writer.add_schema(nlp_schema, "preprocessed squad train dataset")
    valid_writer.write_raw_data(train_datalist)
    valid_writer.commit()
    print("squad dev data write success")
