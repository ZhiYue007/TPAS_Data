from transformers import BertTokenizerhttps://github.com/zhiyuexuejava/tpstudy/blob/main/MTCLN_code/config/fusion_bertConfig.py
from DataManager.Chinese_tokenizer.bert_dataset import BertDataset



class bertConfig():
    def __init__(self, project_dir, MAX_LEN: int = 128):
        self.MAX_LEN = MAX_LEN

        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 64
        self.TEST_BATCH_SIZE = 64
        self.EPOCHS = 50
        self.SEED = 42
        self.lr = 5e-5

        self.DO_TRAIN = True
        self.DO_TEST = True

        self.BERT_PATH = project_dir + "/data/pretrained_models/roberta"
        self.CHINESE_BERT_PATH = project_dir + "/data/pretrained_models/ChineseBERT-base"
        self.TOKENIZER = BertTokenizer.from_pretrained(
            f"{self.BERT_PATH}/vocab.txt", lowercase=True)

        self.pinyinTokenizer = BertDataset(self.CHINESE_BERT_PATH)

        self.pretrain_mode_name = "fusion_bert"

        self.MODEL_SAVE_PATH = project_dir + f"/output/fusion_bert_baseline/"
        self.PREDICT_FILE_SAVE_PATH = project_dir + f"/output"
