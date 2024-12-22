from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def get_real_answer(choices_str, answer_str):
    try:
        return choices_str[answer_str]
    except (ValueError, IndexError, SyntaxError) as e:
        return None

def main():
    dataset = load_dataset("derek-thomas/ScienceQA")
    cols_remove = ["task", "grade", "subject", "category", "skill"]
    dataset = dataset.remove_columns(cols_remove)
    train = dataset['train']
    val = dataset['validation']
    test = dataset['test']

    train = train.to_pandas()
    test = test.to_pandas()
    val = val.to_pandas()

    topics = ['geography', 'biology', 'physics', 'chemistry']

    train = train[train['topic'].isin(topics)]
    test = test[test['topic'].isin(topics)]
    val = val[val['topic'].isin(topics)]

    print(f"Filtered train dataset size: {len(train)}")
    print(f"Filtered test dataset size: {len(test)}")
    print(f"Filtered val dataset size: {len(val)}")

    columns_to_clean = ['image', 'question', 'answer']

    train = train.dropna(subset=columns_to_clean, how='any')
    test = test.dropna(subset=columns_to_clean, how='any')
    val = val.dropna(subset=columns_to_clean, how='any')

    topic_counts_t = train['topic'].value_counts()
    topic_counts_te = test['topic'].value_counts()
    topic_counts_v = val['topic'].value_counts()

    print(topic_counts_t,topic_counts_te,topic_counts_v)
    train = train.groupby('topic', group_keys=False).apply(lambda x: x.sample(min(len(x), 100)))
    train.reset_index(drop=True, inplace=True)

    test = test.groupby('topic', group_keys=False).apply(lambda x: x.sample(min(len(x), 30)))
    test.reset_index(drop=True, inplace=True)

    val = val.groupby('topic', group_keys=False).apply(lambda x: x.sample(min(len(x), 30)))
    val.reset_index(drop=True, inplace=True)

    print (train.shape, test.shape, val.shape)
    train = train.drop(columns=['topic'])
    test = test.drop(columns=['topic'])
    val = val.drop(columns=['topic'])

    train['answer_str']= train.apply(lambda row: get_real_answer(row['choices'], row['answer']), axis=1)
    test['answer_str']= test.apply(lambda row: get_real_answer(row['choices'], row['answer']), axis=1)
    val['answer_str']= val.apply(lambda row: get_real_answer(row['choices'], row['answer']), axis=1)

    train = Dataset.from_pandas(train)
    test = Dataset.from_pandas(test)
    val = Dataset.from_pandas(val)

    datasets = DatasetDict({
    'train': train,
    'test': test,
    'validation': val
    })

    # Push datasets to the Hub
    datasets.push_to_hub('richlukich/scienceQAv1', token='your token')
if __name__ == '__main__':
    main()