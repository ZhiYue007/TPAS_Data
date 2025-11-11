import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from utils.utils import Ortho_algorithm
import torch.nn.functional as F

device = torch.device("cuda:0")


class BertForClozeBaseline(BertPreTrainedModel):

    def __init__(self, config, candidate_num):
        super(BertForClozeBaseline, self).__init__(config)
        self.bert = BertModel(config=config)
        self.candidate_embedding = nn.Embedding(candidate_num, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)



    def forward(self, input_ids, attention_mask, token_type_ids, candidate_ids, positions, candidate_input_ids,pinyin_feature):
        sequence_outputs = self.bert(input_ids, attention_mask, token_type_ids)[0]  # 40 128 768
        positions = positions.unsqueeze(-1).expand(-1, -1, sequence_outputs.size(-1))

        blank_states = torch.gather(sequence_outputs, 1, positions)  # 40 2 768

        batch_size, candidate_group_numbers, candidate_list_numbers = candidate_ids.shape  # 40 2 7

        blank_states_reshaped = blank_states.contiguous().view(batch_size * candidate_group_numbers, -1)  # 80 768

        candidate_reshaped = candidate_ids.contiguous().view(batch_size * candidate_group_numbers, -1)  # 80 7

        candidate_input_ids, candidate_input_mask = candidate_input_ids[:, :, :, 0, :], candidate_input_ids[:, :, :, 1,:]
        candidate_input_ids = candidate_input_ids.contiguous().view(
            batch_size * candidate_group_numbers * candidate_list_numbers,
            -1)
        candidate_input_mask = candidate_input_mask.contiguous().view(
            batch_size * candidate_group_numbers * candidate_list_numbers,
            -1)
        candidate_pooler_output = self.bert(input_ids=candidate_input_ids, attention_mask=candidate_input_mask)[
            'pooler_output']

        candidate_pooler_output = candidate_pooler_output.contiguous().view(batch_size * candidate_group_numbers,
                                                                            candidate_list_numbers, -1)  # 80 7 768


        temp_feature = candidate_pooler_output.mean(dim=1, keepdim=True)
        candidate_projection = Ortho_algorithm(candidate_pooler_output, temp_feature)
        candidate_combination = candidate_projection * 0.3 + candidate_pooler_output * 0.7

        candidate_sim = candidate_combination.contiguous().view(batch_size, candidate_group_numbers,
                                                                candidate_list_numbers, -1)
        encoded_candidate = self.candidate_embedding(candidate_reshaped)  # 80 7 768


        pinyin_feature = self.linear(pinyin_feature)

        candidate_sum = candidate_combination +  encoded_candidate +  pinyin_feature
        # candidate_sum = candidate_combination + 0.5 * encoded_candidate + 0.5 * pinyin_feature
        # candidate_sum = candidate_combination + 2 * encoded_candidate + 2 * pinyin_feature

        multiply_result = torch.einsum('abc,ac->abc', candidate_sum,
                                       blank_states_reshaped)  # 80 7 768

        pooled_output = self.dropout(multiply_result)
        classifier = self.classifier(pooled_output)  # 40 2 7 1
        logits = classifier.view(batch_size, -1, candidate_list_numbers)

        return logits, candidate_sim


class CLIP(nn.Module):
    def __init__(self, hyper, temperature=0.07,ratio=0.8):
        super(CLIP, self).__init__()
        self.hyper = hyper
        self.temperature = torch.tensor(temperature, device=device)
        self.KL_func = nn.KLDivLoss(reduction='batchmean')
        self.ratio = ratio

    def forward(self, candidate_sim, real_label):
        f1, f2 = candidate_sim[:, 0, :, :], candidate_sim[:, 1, :, :]
        sim = F.cosine_similarity(f1.unsqueeze(2), f2.unsqueeze(1), dim=3)
        zero_matrix = torch.zeros_like(sim, device=device)
        one_matrix = torch.ones_like(sim, device=device)
        similarity = torch.max(sim, one_matrix - self.hyper)
        un_similarity = torch.max(sim - self.hyper, zero_matrix)
        soft_label = (one_matrix - real_label) / (sim.shape[0] - 1) * un_similarity + real_label * similarity
        soft_labels = F.softmax(soft_label, dim=1)
        log_sim_output = F.log_softmax(sim / self.temperature, dim=1)
        soft_loss = self.KL_func(log_sim_output, soft_labels)
        hot_loss = self.KL_func(log_sim_output, real_label)
        loss = soft_loss * self.ratio + hot_loss *(1 - self.ratio)
        return loss
