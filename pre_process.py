from tqdm import tqdm
import json
import numpy as np

vocab_dict_path = '/Users/lovever/Downloads/sgns.sogou.bigram'

demo_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/sinanews.demo'
test_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/sinanews.test'
train_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/sinanews.train'

transferred_demo_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/demo.json'
transferred_test_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/test.json'
transferred_train_data_path = '/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/train.json'


def build_vocab_dict(file_path):
    vocab_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            lst = line.split()
            vocab_dict[lst[0]] = lst[1:]
    return vocab_dict


def get_label(counts):
    data = []
    for i in counts:
        cnt = int(i.split(':')[1])
        data.append(cnt)
    return int(np.array(data).argmax())


def match(vocab_dict, text):
    ret = []
    for word in text:
        if word in vocab_dict:
            ret.append(vocab_dict[word])
    return ret


def process_data(vocab_dict, file_path, save_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            lst = line.split()
            info = lst[:10]
            text = lst[10:]
            data.append({
                'id': info[0],
                'label': get_label(info[2:]),
                'text': match(vocab_dict, text)
            })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def main():
    vocab_dict = build_vocab_dict(vocab_dict_path)
    process_data(vocab_dict, demo_data_path, transferred_demo_data_path)
    process_data(vocab_dict, train_data_path, transferred_train_data_path)
    process_data(vocab_dict, test_data_path, transferred_test_data_path)


if __name__ == '__main__':
    main()