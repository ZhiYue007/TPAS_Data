import os
import sys
import torch
import pandas as pd
import pickle
import json
import re
from config.fusion_bertConfig import bertConfig

from tqdm import tqdm

class DataManager(bertConfig):
    def __init__(self, project_dirs, train_path, dev_path, test_path, test_asy_path=None, test_sim_path=None,
                 test_ran_path=None):
        super().__init__(project_dirs)
        self.project_dir = project_dirs
        self.data_dir = os.path.join(project_dirs, "data")

        candidate_vocab = eval(open(self.data_dir + '/all_word_list.txt', encoding="utf-8").readline())
        self.candidate_vocab = {each: i for i, each in enumerate(candidate_vocab)}

        self.text_candidate_ids, self.pinyin_text_candidate_ids, self.pinyin_candidate_ids = self.candidate_to_ids(
            self.candidate_vocab, self.TOKENIZER, self.pinyinTokenizer)

        self.candidate_dict_path = self.data_dir + "/all_summarize.txt"
        front_to_rear_dict, rear_to_front_dict = self.hash_dict(self.candidate_dict_path)
        self.pkl_path = self.data_dir + '/' + self.pretrain_mode_name + '/pkl'
        self.check_dir()
        train_pkl_path = self.pkl_path + '/' + train_path.rsplit(".", 1)[0] + '.pkl'
        dev_pkl_path = self.pkl_path + '/' + dev_path.rsplit(".", 1)[0] + '.pkl'

        test_pkl_path = self.pkl_path + '/' + test_path.rsplit(".", 1)[0] + '.pkl'
        if test_asy_path is not None and test_ran_path is not None and test_sim_path is not None:
            self.test_asy_pkl_path = self.pkl_path + '/' + test_asy_path.rsplit(".", 1)[0] + '.pkl'
            self.test_sim_pkl_path = self.pkl_path + '/' + test_sim_path.rsplit(".", 1)[0] + '.pkl'
            self.test_ran_pkl_path = self.pkl_path + '/' + test_ran_path.rsplit(".", 1)[0] + '.pkl'

        if os.path.exists(train_pkl_path):
            with open(train_pkl_path, 'rb') as file:
                self.train_dataset = pickle.load(file)
            print('data pkl load success, the train num is ', len(self.train_dataset))
        else:
            train_df = self.read_data(os.path.join(self.data_dir, train_path))
            self.train_dataset = ClozeDataset(
                tokenizer=self.TOKENIZER,
                tag=train_df.tag.values,
                data_id=train_df.data_id.values,
                text=train_df.text.values,
                candidate=train_df.candidate.values,
                groundTruth=train_df.groundTruth.values,
                candidate_vocab=self.candidate_vocab,
                text_candidate_id=self.text_candidate_ids,
                pinyin_text_candidate_id=self.pinyin_text_candidate_ids,
                pinyin_candidate_id=self.pinyin_candidate_ids,

                front_to_rear_dict=front_to_rear_dict,
                rear_to_front_dict=rear_to_front_dict,
                max_len=self.MAX_LEN,
            )
            # self.save_data(self.train_dataset, train_pkl_path)
            with open(train_pkl_path, 'wb', ) as file:
                pickle.dump(self.train_dataset, file)
            print('data txt load success, the train num is ', len(self.train_dataset))
        if os.path.exists(dev_pkl_path):
            with open(dev_pkl_path, 'rb') as file:
                self.dev_dataset = pickle.load(file)
            print('data pkl load success, the dev num is ', len(self.dev_dataset))
        else:
            dev_df = self.read_data(os.path.join(self.data_dir, dev_path))
            self.dev_dataset = ClozeDataset(
                tokenizer=self.TOKENIZER,
                tag=dev_df.tag.values,
                data_id=dev_df.data_id.values,
                text=dev_df.text.values,
                candidate=dev_df.candidate.values,
                groundTruth=dev_df.groundTruth.values,
                candidate_vocab=self.candidate_vocab,
                text_candidate_id=self.text_candidate_ids,
                pinyin_text_candidate_id=self.pinyin_text_candidate_ids,
                pinyin_candidate_id=self.pinyin_candidate_ids,
                front_to_rear_dict=front_to_rear_dict,
                rear_to_front_dict=rear_to_front_dict,
                max_len=self.MAX_LEN,
            )
            with open(dev_pkl_path, 'wb', ) as file:
                pickle.dump(self.dev_dataset, file)
            # self.save_data(self.dev_dataset, dev_pkl_path)
            print('data txt load success, the dev num is ', len(self.dev_dataset))
        if os.path.exists(test_pkl_path):
            with open(test_pkl_path, 'rb') as file:
                self.test_dataset = pickle.load(file)
            print('data pkl load success, the test num is ', len(self.test_dataset))
        else:
            test_df = self.read_data(os.path.join(self.data_dir, test_path))
            self.test_dataset = ClozeDataset(
                tokenizer=self.TOKENIZER,
                tag=test_df.tag.values,
                data_id=test_df.data_id.values,
                text=test_df.text.values,
                candidate=test_df.candidate.values,
                groundTruth=test_df.groundTruth.values,
                candidate_vocab=self.candidate_vocab,
                text_candidate_id=self.text_candidate_ids,
                pinyin_text_candidate_id=self.pinyin_text_candidate_ids,
                pinyin_candidate_id=self.pinyin_candidate_ids,
                front_to_rear_dict=front_to_rear_dict,
                rear_to_front_dict=rear_to_front_dict,
                max_len=self.MAX_LEN,
            )
            with open(test_pkl_path, 'wb', ) as file:
                pickle.dump(self.test_dataset, file)
            # self.save_data(self.test_dataset, test_pkl_path)
            print('data txt load success, the test num is ', len(self.test_dataset))
        if test_asy_path != None:
            if os.path.exists(self.test_asy_pkl_path):
                with open(self.test_asy_pkl_path, 'rb') as file:
                    self.test_asy_dataset = pickle.load(file)
                print('data pkl load success, the test asy num is ', len(self.test_asy_dataset))
            else:
                test_asy_df = self.read_data(os.path.join(self.data_dir, test_asy_path))
                self.test_asy_dataset = ClozeDataset(
                    tokenizer=self.TOKENIZER,
                    tag=test_asy_df.tag.values,
                    data_id=test_asy_df.data_id.values,
                    text=test_asy_df.text.values,
                    candidate=test_asy_df.candidate.values,
                    groundTruth=test_asy_df.groundTruth.values,
                    candidate_vocab=self.candidate_vocab,
                    text_candidate_id=self.text_candidate_ids,
                    pinyin_text_candidate_id=self.pinyin_text_candidate_ids,
                    pinyin_candidate_id=self.pinyin_candidate_ids,
                    front_to_rear_dict=front_to_rear_dict,
                    rear_to_front_dict=rear_to_front_dict,
                    max_len=self.MAX_LEN,
                )
                with open(self.test_asy_pkl_path, 'wb', ) as file:
                    pickle.dump(self.test_asy_dataset, file)
                # self.save_data(self.test_asy_dataset, self.test_asy_pkl_path)
                print('data txt load success, the test asy num is ', len(self.test_asy_dataset))
        if test_sim_path != None:
            if os.path.exists(self.test_sim_pkl_path):
                with open(self.test_sim_pkl_path, 'rb') as file:
                    self.test_sim_dataset = pickle.load(file)
                print('data txt load success, the test sim num is ', len(self.test_sim_dataset))
            else:
                test_sim_df = self.read_data(os.path.join(self.data_dir, test_sim_path))
                self.test_sim_dataset = ClozeDataset(
                    tokenizer=self.TOKENIZER,
                    tag=test_sim_df.tag.values,
                    data_id=test_sim_df.data_id.values,
                    text=test_sim_df.text.values,
                    candidate=test_sim_df.candidate.values,
                    groundTruth=test_sim_df.groundTruth.values,
                    candidate_vocab=self.candidate_vocab,
                    text_candidate_id=self.text_candidate_ids,
                    pinyin_text_candidate_id=self.pinyin_text_candidate_ids,
                    pinyin_candidate_id=self.pinyin_candidate_ids,

                    front_to_rear_dict=front_to_rear_dict,
                    rear_to_front_dict=rear_to_front_dict,
                    max_len=self.MAX_LEN,
                )
                with open(self.test_sim_pkl_path, 'wb', ) as file:
                    pickle.dump(self.test_sim_dataset, file)
                # self.save_data(self.test_sim_dataset, self.test_sim_pkl_path)
                print('data txt load success, the test sim num is ', len(self.test_sim_dataset))
        if test_ran_path != None:
            if os.path.exists(self.test_ran_pkl_path):
                with open(self.test_ran_pkl_path, 'rb') as file:
                    self.test_ran_dataset = pickle.load(file)
                print('data txt load success, the test ran num is ', len(self.test_ran_dataset))
            else:
                test_ran_df = self.read_data(os.path.join(self.data_dir, test_ran_path))
                self.test_ran_dataset = ClozeDataset(
                    tokenizer=self.TOKENIZER,
                    tag=test_ran_df.tag.values,
                    data_id=test_ran_df.data_id.values,
                    text=test_ran_df.text.values,
                    candidate=test_ran_df.candidate.values,
                    groundTruth=test_ran_df.groundTruth.values,
                    candidate_vocab=self.candidate_vocab,
                    text_candidate_id=self.text_candidate_ids,
                    pinyin_text_candidate_id = self.pinyin_text_candidate_ids,
                    pinyin_candidate_id=self.pinyin_candidate_ids,

                    front_to_rear_dict=front_to_rear_dict,
                    rear_to_front_dict=rear_to_front_dict,
                    max_len=self.MAX_LEN,
                )
                with open(self.test_ran_pkl_path, 'wb', ) as file:
                    pickle.dump(self.test_ran_dataset, file)
                # self.save_data(self.test_ran_dataset, self.test_ran_pkl_path)
                print('data txt load success, the test ran num is ', len(self.test_ran_dataset))


    def save_data(self, dataset, save_path):
        data = []
        tk = tqdm(dataset, total=len(dataset))
        for bi, item in enumerate(tk):
            data.append(item)
        with open(save_path, 'wb') as file:
            pickle.dump(data, file)

    def hash_dict(self, candidate_dict_path):
        front_to_rear_dict = {}
        rear_to_front_dict = {}
        with open(candidate_dict_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            nul, numbers = first_line.split()
            for _ in range(int(numbers)):
                line = file.readline().strip().split(" ", 1)
                front, rear = str(line[1]).split("——")
                if front not in front_to_rear_dict:
                    front_to_rear_dict[front] = [rear]
                else:
                    front_to_rear_dict[front].append(rear)
                if rear not in rear_to_front_dict:
                    rear_to_front_dict[rear] = [front]
                else:
                    rear_to_front_dict[rear].append(front)
        return front_to_rear_dict, rear_to_front_dict

    def check_dir(self):
        if not os.path.exists(self.MODEL_SAVE_PATH):
            os.makedirs(self.MODEL_SAVE_PATH)
        if not os.path.exists(self.PREDICT_FILE_SAVE_PATH):
            os.makedirs(self.PREDICT_FILE_SAVE_PATH)
        if not os.path.exists(self.pkl_path):
            os.makedirs(self.pkl_path)

    def candidate_to_ids(self, candidate_vocab, text_tokenizer, pinyin_tokenizer, candidate_max_len=24):

        text_candidate_input_ids = []
        pinyin_text_candidate_input_ids = []
        pinyin_candidate_input_ids = []

        for key, value in enumerate(candidate_vocab):
            text_value = text_tokenizer.encode(value, add_special_tokens=True)
            text_masks = [1] * len(text_value) + [0] * (candidate_max_len - len(text_value))
            text_value = text_value + (candidate_max_len - len(text_value)) * [0]
            text_candidate_input_ids.append([text_value, text_masks])

            input_ids_ori, pinyin_ids_ori = pinyin_tokenizer.encode(value, add_special_tokens=True)
            text_masks = [1] * len(input_ids_ori) + [0] * (candidate_max_len - len(input_ids_ori))
            text_value = input_ids_ori + (candidate_max_len - len(input_ids_ori)) * [0]

            pinyin_value = pinyin_ids_ori + (candidate_max_len - len(input_ids_ori)) * 8 * [0]

            pinyin_text_candidate_input_ids.append([text_value, text_masks])
            pinyin_candidate_input_ids.append(pinyin_value)


        return text_candidate_input_ids,pinyin_text_candidate_input_ids, pinyin_candidate_input_ids

    def read_data(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as fin:
            data_id = 10000000
            for line in fin.readlines():
                cur_data = json.loads(line)
                groundTruth = cur_data["groundTruth"]
                candidates = cur_data["candidates"]

                content = cur_data["content"]
                realCount = cur_data["realCount"]
                for i in range(realCount):
                    content = content.replace("#mask#", f"#mask{i + 1}#", 1)
                tags = re.findall("#mask\d+#", content)

                data.append({
                    "data_id": data_id,
                    "tag": tags,
                    "text": content,
                    "candidate": candidates,
                    "groundTruth": groundTruth
                })
                data_id += 1
        df_data = pd.DataFrame(data)
        return df_data

class ClozeDataset:

    def __init__(self, tokenizer, tag, data_id, text, candidate, groundTruth,
                 candidate_vocab,text_candidate_id, pinyin_text_candidate_id, pinyin_candidate_id,
                 front_to_rear_dict, rear_to_front_dict,max_len=128):

        self.tag = tag
        self.data_id = data_id
        self.text = text
        self.candidate = candidate
        self.groundTruth = groundTruth
        self.candidate_vocab = candidate_vocab

        self.tokenizer = tokenizer
        self.text_candidate_ids = text_candidate_id,
        self.pinyin_text_candidate_id = pinyin_text_candidate_id,
        self.pinyin_candidate_ids = pinyin_candidate_id,

        self.front_to_rear_dict = front_to_rear_dict
        self.rear_to_front_dict = rear_to_front_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        feature_id = int(self.tag[item][0][5: -1])
        left_part, right_part = re.split(self.tag[item][0], self.text[item])
        middle_part, right_part = re.split(self.tag[item][1], right_part)
        text_left_ids = self.tokenizer.encode(left_part, add_special_tokens=False)
        text_middle_ids = self.tokenizer.encode(middle_part, add_special_tokens=False)
        text_right_ids = self.tokenizer.encode(right_part, add_special_tokens=False)
        text_ids = text_left_ids + [self.tokenizer.mask_token_id] + text_middle_ids + [
            self.tokenizer.mask_token_id] + text_right_ids
        text_input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]  # 前面后面加上加上开头和结尾
        text_positions = [i for i, token_id in enumerate(text_input_ids) if token_id == self.tokenizer.mask_token_id]
        text_token_type_ids = [0] * len(text_input_ids) + [0] * (self.max_len - len(text_input_ids))  # 0表示前后都是一句话
        text_input_masks = [1] * len(text_input_ids) + [0] * (self.max_len - len(text_input_ids))  # 1表示是话，后面是填充的无效部分
        text_input_ids = text_input_ids + [0] * (self.max_len - len(text_input_ids))  # 不够长的位置填充0


        text_candidate_ids_input = []
        pinyin_text_candidate_ids_input = []
        pinyin_candidate_ids_input = []
        label_list = []
        candidate_ids_list = []
        for i in range(len(self.candidate[item])):
            label = self.candidate[item][i].index(self.groundTruth[item][i])
            label_list.append(label)
            candidate_ids = [self.candidate_vocab[each] for each in
                             self.candidate[item][i]]
            text_candidate_ids = [self.text_candidate_ids[0][each] for each in
                                  candidate_ids]
            pinyin_text_candidate_id = [self.pinyin_text_candidate_id[0][each] for each in
                                    candidate_ids]
            pinyin_candidate_id = [self.pinyin_candidate_ids[0][each] for each in
                                    candidate_ids]
            text_candidate_ids_input.append(text_candidate_ids)
            pinyin_text_candidate_ids_input.append(pinyin_text_candidate_id)
            pinyin_candidate_ids_input.append(pinyin_candidate_id)
            candidate_ids_list.append(candidate_ids)

        front_list = self.candidate[item][0]
        rear_list = self.candidate[item][1]
        hot_matrix = torch.zeros(7, 7, dtype=torch.float32)
        for index, value in enumerate(front_list):
            if value not in self.front_to_rear_dict:
                continue
            to_rear = self.front_to_rear_dict[value]
            for i in range(len(to_rear)):
                if to_rear[i] in rear_list:
                    rear_index = rear_list.index(to_rear[i])
                    hot_matrix[index, rear_index] = 1
        for index, value in enumerate(rear_list):
            if value not in self.rear_to_front_dict:
                continue
            to_front = self.rear_to_front_dict[value]
            for i in range(len(to_front)):
                if to_front[i] in front_list:
                    front_index = front_list.index(to_front[i])
                    hot_matrix[front_index, index] = 1

        assert len(text_input_ids) == self.max_len
        assert len(text_input_masks) == self.max_len
        assert len(text_token_type_ids) == self.max_len


        return {
            'data_id': torch.tensor(self.data_id[item], dtype=torch.long),
            'feature_id': torch.tensor(feature_id, dtype=torch.long),
            'text_input_ids': torch.tensor(text_input_ids, dtype=torch.long),
            'text_input_masks': torch.tensor(text_input_masks, dtype=torch.long),
            'text_token_type_ids': torch.tensor(text_token_type_ids, dtype=torch.long),
            'text_position': torch.tensor(text_positions, dtype=torch.long),

            'text_candidate_ids': torch.tensor(text_candidate_ids_input, dtype=torch.long),
            'pinyin_text_candidate_ids': torch.tensor(pinyin_text_candidate_ids_input, dtype=torch.long),
            'pinyin_candidate_ids': torch.tensor(pinyin_candidate_ids_input, dtype=torch.long).view(2,7,-1,8),

            'candidate_ids': torch.tensor(candidate_ids_list, dtype=torch.long),
            'label': torch.tensor(label_list, dtype=torch.long),
            'hot_matrix': hot_matrix

        }


