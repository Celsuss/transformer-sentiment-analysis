"""Main function of the training script."""
from torch.utils.data.dataset import random_split
from datasets import load_dataset
import torch


def load_data():
    """Load data using torchtext.datasets."""
    ds = load_dataset("stanfordnlp/imdb")
    print(f'Downloaded imdb dataset {ds}')
    train = ds['train']
    test = ds['test']

    # test_size = test['num_rows'] * 0.8
    # valid_size = test['num_rows'] * 0.2
    # train_dataset, valid_dataset = random_split(
    #     list(train_dataset_raw), [train_size, valid_size]
    # )
    # return train_dataset, valid_dataset, test_dataset_raw
    return train, None, test


def main():
    """Master function of the training script."""
    torch.manual_seed(1)
    tran, validation, test = load_data()
    print('done')
    return


if __name__ == '__main__':
    main()
