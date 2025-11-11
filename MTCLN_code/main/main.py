import os
import random
import sys
import logging
import torch
import numpy as np

sys.path.append('../')

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup  # WarmupLinearSchedule
from tqdm import tqdm
from utils.utils import EarlyStopping, acc_calculate
from model.fusion_kl_model import BertForClozeBaseline,CLIP
from model.chinese_bert.modeling_glycebert import GlyceBertModel
from DataManager.fusion_dataManager import DataManager

workdir = os.getcwd()
project_dir = os.path.split(workdir)[0]
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
train_path = "output_all_train_data.txt"
test_path = "output_all_test_data.txt"
dev_path = "output_all_dev_data.txt"
test_asy_path = "output_all_test_asy_data.txt"
test_ran_path = "output_all_test_ran_data.txt"
test_sim_path = "output_all_test_sim_data.txt"
dataManager = DataManager(project_dir, train_path, dev_path, test_path,test_asy_path=test_asy_path,
                          test_sim_path=test_sim_path,test_ran_path=test_ran_path)
dataManager.MODEL_SAVE_PATH += os.path.basename(__file__).split(".")[0]


def run_one_step(batch, pinyin_cls, model, device):
    input_ids = batch["text_input_ids"].to(device)
    input_mask = batch["text_input_masks"].to(device)
    token_type_ids = batch["text_token_type_ids"].to(device)
    position = batch["text_position"].to(device)
    candidate_ids = batch["candidate_ids"].to(device)
    text_candidate_inputs_ids = batch["text_candidate_ids"].to(device)

    logits = model(
        input_ids = input_ids,
        attention_mask = input_mask,
        token_type_ids=token_type_ids,
        candidate_ids=candidate_ids,
        positions=position,
        candidate_input_ids=text_candidate_inputs_ids,
        pinyin_feature = pinyin_cls

    )
    return logits

def chinese_bert_run_one_step(batch, model, device):
    pinyin_text_candidate_ids = batch["pinyin_text_candidate_ids"].to(device)
    pinyin_candidate_ids = batch["pinyin_candidate_ids"].to(device)

    candidate_input_ids, candidate_input_mask = pinyin_text_candidate_ids[:, :, :, 0, :], \
        pinyin_text_candidate_ids[:, :, :, 1, :]
    batch_size, candidate_group_numbers, candidate_list_numbers,_ = candidate_input_ids.shape
    candidate_input_ids = candidate_input_ids.contiguous().view(
        batch_size * candidate_group_numbers * candidate_list_numbers,
        -1)
    candidate_input_mask = candidate_input_mask.contiguous().view(
        batch_size * candidate_group_numbers * candidate_list_numbers,
        -1)
    candidate_pinyin_input_mask = pinyin_candidate_ids.contiguous().view(
        batch_size * candidate_group_numbers * candidate_list_numbers,
        -1, 8)
    pinyin_cls = model(candidate_input_ids, candidate_pinyin_input_mask,
                                   attention_mask=candidate_input_mask)[1]
    pinyin_cls = pinyin_cls.contiguous().view(batch_size * candidate_group_numbers,
                                              candidate_list_numbers, -1)
    return pinyin_cls


def train_fn(data_loader, model,chinese_bert,clip_model, optimizer, device, epoch, scheduler=None):
    """
    Trains the bert model on the twitter data
    """
    # Set model to training mode (dropout + sampled batch norm is activated)
    model.train()

    # Set tqdm to add loading screen and set the length
    tk0 = tqdm(data_loader, total=len(data_loader))
    # Train the model on each batch
    # Reset gradients
    pred_labels = []
    true_labels = []
    all_loss = 0
    all_loss0 = 0
    all_loss1 = 0
    iteration_count = 0
    for bi, batch in enumerate(tk0):
        model.zero_grad()
        pinyin_cls = chinese_bert_run_one_step(batch, chinese_bert, device)
        logits, candidate_sim = run_one_step(batch, pinyin_cls, model, device)  # 40 2 7

        hot_label = batch["hot_matrix"].to(device)
        clip_loss = clip_model(candidate_sim, hot_label)

        label = batch["label"]
        logits_0 = logits[:, 0, :].to(device)
        logits_1 = logits[:, 1, :].to(device)

        label_0 = label[:, 0].to(device)
        label_1 = label[:, 1].to(device)
        # label = batch["label"].to(device)  # 40 2

        # # Calculate batch loss based on CrossEntropy
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_0 = loss_fn(logits_0, label_0)
        loss_1 = loss_fn(logits_1, label_1)
        loss = loss_0 + loss_1
        all_loss += loss.item()
        all_loss0 += loss_0.item()
        all_loss1 += loss_1.item()
        iteration_count += len(batch)
        all_clip_loss = clip_loss + loss
        all_clip_loss.backward()
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()
        # Update scheduler
        #   # 更新learning rate
        # print(f"lr={scheduler.get_lr()}")
        scheduler.step()

        # Calculate the acc score based on the predictions for this batch
        outputs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        pred_label = np.argmax(outputs, axis=-1)
        acc = acc_calculate(label.cpu().numpy(), pred_label)
        pred_labels.extend(pred_label.tolist())
        true_labels.extend(label.cpu().numpy().tolist())
        # Print the average loss and jaccard score at the end of each batch
        tk0.set_postfix(epoch=epoch, acc=acc, loss=loss.item())

    total_acc = acc_calculate(np.array(true_labels), np.array(pred_labels))
    all_loss = all_loss / iteration_count
    all_loss0 = all_loss0 / iteration_count
    all_loss1 = all_loss1 / iteration_count
    logger.info(
        f"train_epoch: {epoch + 1}, acc = {total_acc} , loss = {all_loss}, loss0 = {all_loss0}, loss1 ={all_loss1}")


def eval_fn(valid_data_loader, model,chinese_bert, device):
    """
    Evaluation function to predict on the test set
    """
    # Set model to evaluation mode
    model.eval()
    pred_labels = []
    true_labels = []
    all_loss = 0
    all_loss0 = 0
    all_loss1 = 0
    iteration_count = 0

    with torch.no_grad():

        tk0 = tqdm(valid_data_loader, total=len(valid_data_loader))
        # Make predictions and calculate loss / acc, f1 score for each batch
        for bi, batch in enumerate(tk0):
            # Use ids, masks, and token types as input to the model
            # Predict logits for each of the input tokens for each batch
            pinyin_cls = chinese_bert_run_one_step(batch, chinese_bert, device)
            logits, candidate_sim = run_one_step(batch, pinyin_cls, model, device)  # 40 2 7


            label = batch["label"]
            logits_0 = logits[:, 0, :].to(device)
            logits_1 = logits[:, 1, :].to(device)

            label_0 = label[:, 0].to(device)
            label_1 = label[:, 1].to(device)
            # label = batch["label"].to(device)  # 40 2

            # # Calculate batch loss based on CrossEntropy
            loss_fn = torch.nn.CrossEntropyLoss()
            loss_0 = loss_fn(logits_0, label_0)
            loss_1 = loss_fn(logits_1, label_1)
            loss = loss_0 + loss_1
            all_loss += loss.item()
            all_loss0 += loss_0.item()
            all_loss1 += loss_1.item()
            iteration_count += len(batch)

            outputs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=-1)
            acc = acc_calculate(label.cpu().numpy(), pred_label)
            pred_labels.extend(pred_label.tolist())
            true_labels.extend(label.cpu().numpy().tolist())
            # Print the running average loss and acc
            tk0.set_postfix(loss=loss.item(), acc=acc)

    total_acc = acc_calculate(np.array(true_labels), np.array(pred_labels))
    all_loss = all_loss / iteration_count
    all_loss0 = all_loss0 / iteration_count
    all_loss1 = all_loss1 / iteration_count
    logger.info(
        f"dev: acc = {total_acc} , loss = {all_loss}, loss0 = {all_loss0}, loss1 ={all_loss1}")
    return total_acc


def train():
    train_dataset = dataManager.train_dataset
    dev_dataset = dataManager.dev_dataset
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=dataManager.TRAIN_BATCH_SIZE,
        num_workers=0
    )
    valid_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=dataManager.VALID_BATCH_SIZE,
        num_workers=0
    )
    model_config = transformers.BertConfig.from_pretrained(dataManager.BERT_PATH)
    # This is important to set since we want to concatenate the hidden states from the last 2 BERT layers
    model_config.output_hidden_states = True


    # model = BertForClozeBaseline.from_pretrained(path1=dataManager.BERT_PATH,path2=dataManager.CHINESE_BERT_PATH,
    #                              config=model_config, candidate_num=len(dataManager.candidate_vocab))
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=dataManager.BERT_PATH,
                                                 config=model_config, candidate_num=len(dataManager.candidate_vocab))
    # Move the model to the GPU
    model.to(device)

    chinese_bert = GlyceBertModel.from_pretrained(dataManager.CHINESE_BERT_PATH)
    chinese_bert.to(device)

    clip_model = CLIP(hyper=0.2)
    clip_model.to(device)
    # Calculate the number of training steps
    num_train_warm_steps = int(len(train_dataset) / dataManager.TRAIN_BATCH_SIZE * dataManager.EPOCHS) * 0.05
    num_train_steps = int(len(train_dataset) / dataManager.TRAIN_BATCH_SIZE * dataManager.EPOCHS)
    # Get the list of named parameters
    print(num_train_steps)
    # print(num_train_steps)
    param_optimizer = list(model.named_parameters())
    # Specify parameters where weight decay shouldn't be applied
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # Instantiate AdamW optimizer with our two sets of parameters, and a learning rate of 3e-5
    optimizer = AdamW(optimizer_parameters, lr=dataManager.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_warm_steps,
        num_training_steps=num_train_steps,

    )
    # Apply early stopping with patience of 2
    # This means to stop training new epochs when 2 rounds have passed without any improvement
    es = EarlyStopping(patience=8, mode="max", delta=0.00001)
    # I'm training only for 3 epochs even though I specified 5!!!
    for epoch in range(dataManager.EPOCHS):
        train_fn(train_data_loader, model,chinese_bert,clip_model, optimizer, device, epoch + 1, scheduler)
        eval_acc = eval_fn(valid_data_loader, model,chinese_bert, device)
        logger.info(f"epoch: {epoch + 1}, acc = {eval_acc}")
        es(epoch, eval_acc, model, model_path=dataManager.MODEL_SAVE_PATH)
        if es.early_stop:
            print("********** Early stopping ********")
            break


def predict(dataset):
    test_dataset = dataset
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=dataManager.TEST_BATCH_SIZE
    )
    # Load pretrained BERT (bert-base-uncased)
    model_path = dataManager.MODEL_SAVE_PATH
    model_config = transformers.BertConfig.from_pretrained(model_path)
    # Instantiate our model with `model_config`
    model = BertForClozeBaseline.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config, candidate_num=len(dataManager.candidate_vocab))
    # # Load each of the five trained models and move to GPU
    trained_model_path = os.path.join(model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(trained_model_path))

    model.to(device)

    chinese_bert = GlyceBertModel.from_pretrained(dataManager.CHINESE_BERT_PATH)
    chinese_bert.to(device)

    model.eval()

    pred_labels = []
    true_labels = []
    # Turn of gradient calculations
    with torch.no_grad():
        tk0 = tqdm(test_dataloader, total=len(test_dataloader))
        # Predict the span containing the sentiment for each batch
        for bi, batch in enumerate(tk0):
            # Predict logits
            pinyin_cls = chinese_bert_run_one_step(batch, chinese_bert, device)
            logits, candidate_sim = run_one_step(batch, pinyin_cls, model, device)  # 40 2 7
            label = batch["label"]
            outputs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
            pred_label = np.argmax(outputs, axis=-1)
            pred_labels.extend(pred_label.tolist())
            true_labels.extend(label.cpu().numpy().tolist())
    test_acc = acc_calculate(np.array(true_labels), np.array(pred_labels))
    logger.info(f"test acc: {test_acc}")
    return pred_labels


def seed_set(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run():
    seed_set(dataManager.SEED)
    if dataManager.DO_TRAIN:
        train()
    if dataManager.DO_TEST:
        print("test_acc")
        predict(dataManager.test_dataset)
        print("test_asy_acc")
        predict(dataManager.test_asy_dataset)
        print("test_sim_acc")
        predict(dataManager.test_sim_dataset)
        print("test_ran_acc")
        predict(dataManager.test_ran_dataset)

if __name__ == '__main__':
    run()
