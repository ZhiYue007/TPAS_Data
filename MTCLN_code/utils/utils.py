import numpy as np
import pickle
import os
import torch


def load_pkl_data(filePath):
    with open(filePath, 'rb') as fp:
        data_pkl = fp.read()
    print(f'loaded {filePath}')
    return pickle.loads(data_pkl)


def save_pkl_data(data, filePath):
    data_pkl = pickle.dumps(data)
    with open(filePath, 'wb') as fp:
        fp.write(data_pkl)
    print(f'saved {filePath}')


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.best_threshold = 0.0
        self.delta = delta
        self.epoch = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(model_path, safe_serialization=False)
        self.val_score = epoch_score


def acc_calculate(predict_label, label):
    correct_predictions = np.all(predict_label == label, axis=1)
    accuracy = np.mean(correct_predictions)
    return accuracy


class Matrix(object):
    def __init__(self, vectors):
        self.vectors = vectors
        self.batch_size = vectors[0]
        self.num_samples = vectors.shape[1]
        self.dimension = vectors.shape[2]

    def __str__(self):
        return self.vectors

    def normalized(self):
        magnitude = torch.sqrt(torch.sum(torch.pow(self.vectors, 2), dim=-1)).unsqueeze(-1)
        return self.vectors / magnitude

    def component_parallel_to(self, basis):
        u = basis.normalized()  # 标准化 b, d
        weight = torch.sum(self.vectors * u, dim=-1, keepdim=True)
        return u * weight

    def component_orthogonal_to(self, bais):
        projection = self.component_parallel_to(bais)
        return self.vectors - projection


def common_algorithm(original_feature, trivial_feature):
    original_feature = Matrix(original_feature)
    trivial_feature = Matrix(trivial_feature)
    d = original_feature.component_parallel_to(trivial_feature)

    return d


def Ortho_algorithm(original_feature, trivial_feature):
    original_feature = Matrix(original_feature)
    trivial_feature = Matrix(trivial_feature)
    d = original_feature.component_orthogonal_to(trivial_feature)
    d = Matrix(d)
    f = original_feature.component_parallel_to(d)
    return f
