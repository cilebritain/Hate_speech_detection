'''import fastNLP'''
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Tester
from fastNLP import Vocabulary
from fastNLP.io import DataBundle
from fastNLP.embeddings import BertEmbedding
from fastNLP.models import BertForSequenceClassification
from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric, Adam

'''import other package'''
import os
import csv
import torch

'''load the csv'''
file2label = {}
with open('dataset/annotations_metadata.csv') as f:
    for line in f:
        items = line.split(',')
        file2label[items[0]+".txt"] = (1 if items[4].split('\n')[0] == 'hate' else 0)


'''load the Hate-Speech data'''
train_dataset = DataSet()
dev_dataset = DataSet()
test_dataset = DataSet()
cnt = 0
length = len(os.listdir("dataset/all_files"))

for file in os.listdir("dataset/all_files"):
    with open("dataset/all_files/"+file) as f:
        cnt += 1
        raw_words = f.read()
        words = raw_words.split()
        seq_len = len(words)
        if cnt > length * 0.9:
            test_dataset.append(Instance(raw_words = raw_words,
                                    words = words,
                                    seq_len = seq_len,
                                    target = file2label[file]))
        elif cnt > length * 0.8:
            dev_dataset.append(Instance(raw_words = raw_words,
                        words = words,
                        seq_len = seq_len,
                        target = file2label[file]))
        else:
            train_dataset.append(Instance(raw_words = raw_words,
                        words = words,
                        seq_len = seq_len,
                        target = file2label[file]))

train_dataset.set_input('words', 'seq_len', 'target')
dev_dataset.set_input('words', 'seq_len', 'target')
test_dataset.set_input('words', 'seq_len', 'target')

train_dataset.set_target('target')
dev_dataset.set_target('target')
test_dataset.set_target('target')

'''build vocabulary'''
vocab = Vocabulary()
vocab.from_dataset(train_dataset, field_name='words', no_create_entry_dataset=[dev_dataset, test_dataset])
vocab.index_dataset(train_dataset, dev_dataset, test_dataset, field_name='words')

target_vocab = Vocabulary(padding=None, unknown=None)
target_vocab.from_dataset(train_dataset, field_name='target', no_create_entry_dataset=[dev_dataset, test_dataset])
target_vocab.index_dataset(train_dataset, dev_dataset, test_dataset, field_name='target')

'''build bundle'''
data_dict = {"train":train_dataset, "dev":dev_dataset, "test":test_dataset}
vocab_dict = {"words":vocab, "target":target_vocab}
data_bundle = DataBundle(vocab_dict, data_dict)
print(data_bundle)

'''build model'''
embed = BertEmbedding(data_bundle.get_vocab('words'), model_dir_or_name='en-base-uncased', include_cls_sep=True)
model = BertForSequenceClassification(embed, len(data_bundle.get_vocab('target')))
# model = BertForSequenceClassification(embed, 2)

device = 0 if torch.cuda.is_available() else 'cpu'
trainer = Trainer(data_bundle.get_dataset('train'), model,
                  optimizer=Adam(model_params=model.parameters(), lr=2e-5),
                  loss=CrossEntropyLoss(target='target'), device=device,
                  batch_size=8, dev_data=data_bundle.get_dataset('dev'),
                  metrics=AccuracyMetric(target='target'), n_epochs=2, print_every=1)
trainer.train()

tester = Tester(data_bundle.get_dataset('test'), model, batch_size=128, metrics=AccuracyMetric())
tester.test()


