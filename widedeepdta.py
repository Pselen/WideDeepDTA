import json
import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from utils import tokenize_with_hf, encode_smiles, strip_path
from evaluation import evaluate_predictions

class WideDeepDTA:
    def __init__(self, max_smi_len=100, max_prot_len=1000,
                 seq_embedding_dim=32,
                 pretrained_embedding_dim=32,
                 learning_rate=0.001,
                 batch_size=256, n_epochs=200, # 256, 200
                 num_filters=32, smi_filter_len=4, prot_filter_len=6):
        print('DeepDTA: Building model')
        self.max_smi_len = max_smi_len
        self.max_prot_len = max_prot_len
        self.seq_embedding_dim = seq_embedding_dim
        self.pretrained_embedding_dim = pretrained_embedding_dim
        self.num_filters = num_filters
        self.smi_filter_len = smi_filter_len
        self.prot_filter_len = prot_filter_len

        self.chem_vocab_size = 94
        self.prot_vocab_size = 26
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.history = {}
        self.model = self.build()
        print('DeepDTA: Model compiled')

    def from_file(cls, path):
        with open(f'{path}/params.json') as f:
            dct = json.load(f)

        instance = cls(**dct)

        instance.model = tf.keras.models.load_model(f'{path}/model')

        with open(f'{path}/history.json') as f:
            instance.history = json.load(f)
        return instance

    def build(self):
        # Inputs
        chemical_seqs = Input(shape=(self.max_smi_len,), dtype='int32')

        # chemical representation
        chemical_seq_representation = Embedding(input_dim=self.chem_vocab_size + 1,
                                                output_dim=self.seq_embedding_dim,
                                                input_length=self.max_smi_len,
                                                mask_zero=True)(chemical_seqs)
        chemical_seq_representation = Conv1D(filters=self.num_filters,
                                             kernel_size=self.smi_filter_len,
                                             activation='relu',
                                             padding='valid',
                                             strides=1)(chemical_seq_representation)
        chemical_seq_representation = Conv1D(filters=self.num_filters * 2,
                                             kernel_size=self.smi_filter_len,
                                             activation='relu',
                                             padding='valid',
                                             strides=1)(chemical_seq_representation)
        chemical_seq_representation = Conv1D(filters=self.num_filters * 3,
                                             kernel_size=self.smi_filter_len,
                                             activation='relu',
                                             padding='valid',
                                             strides=1)(chemical_seq_representation)
        chemical_seq_representation = GlobalMaxPooling1D()(chemical_seq_representation)

        # Protein representation
        protein_seqs = Input(shape=(self.max_prot_len,), dtype='int32')
        protein_seq_representation = Embedding(input_dim=self.prot_vocab_size + 1,
                                               output_dim=self.seq_embedding_dim,
                                               input_length=self.max_prot_len,
                                               mask_zero=True)(protein_seqs)
        protein_seq_representation = Conv1D(filters=self.num_filters,
                                            kernel_size=self.prot_filter_len,
                                            activation='relu',
                                            padding='valid',
                                            strides=1)(protein_seq_representation)
        protein_seq_representation = Conv1D(filters=self.num_filters * 2,
                                            kernel_size=self.prot_filter_len,
                                            activation='relu',
                                            padding='valid',
                                            strides=1)(protein_seq_representation)
        protein_seq_representation = Conv1D(filters=self.num_filters * 3,
                                            kernel_size=self.prot_filter_len,
                                            activation='relu',
                                            padding='valid',
                                            strides=1)(protein_seq_representation)
        protein_seq_representation = GlobalMaxPooling1D()(protein_seq_representation)

        pretrained_chemical_emb = Input(
            shape=(self.pretrained_embedding_dim), dtype='float64')
        pretrained_prot_emb = Input(
            shape=(self.pretrained_embedding_dim), dtype='float64')

        interaction_representation = Concatenate(axis=-1)([chemical_seq_representation,
                                                           pretrained_chemical_emb,
                                                           protein_seq_representation,
                                                           pretrained_prot_emb])

        # Fully connected layers
        FC1 = Dense(1024, activation='relu')(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation='relu')(FC1)
        FC2 = Dropout(0.1)(FC2)
        FC3 = Dense(512, activation='relu')(FC2)
        predictions = Dense(1, kernel_initializer='normal')(FC3)

        opt = Adam(self.learning_rate)
        widedeepdta = Model(
            inputs=[chemical_seqs,
                    pretrained_chemical_emb,
                    protein_seqs,
                    pretrained_prot_emb], outputs=[predictions])
        widedeepdta.compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])
        return widedeepdta

    def vectorize_chemicals(self, chemicals):
        encoded_smiles = [encode_smiles(smiles) for smiles in chemicals]
        return np.array(tokenize_with_hf('chemical/chembl27_enc_94',
                                         encoded_smiles,
                                         padding_len=self.max_smi_len,
                                         out_type='int'))

    def vectorize_proteins(self, aa_sequences):
        return np.array(tokenize_with_hf('protein/uniprot_26',
                                         aa_sequences,
                                         padding_len=self.max_prot_len,
                                         out_type='int'))

    def train(self, train_chemicals, train_chemical_embs, train_proteins, train_protein_embs, train_labels,
              val_chemicals=None, val_chemical_embs=None, val_proteins=None, val_protein_embs=None, val_labels=None):

        train_chemical_seq_vectors = self.vectorize_chemicals(train_chemicals)
        train_protein_seq_vectors = self.vectorize_proteins(train_proteins)
        train_labels = np.array(train_labels)

        val_tuple = None
        if val_chemicals is not None and val_proteins is not None and val_labels is not None:
            val_chemical_seq_vectors = self.vectorize_chemicals(val_chemicals)
            val_protein_seq_vectors = self.vectorize_proteins(val_proteins)
            val_tuple = ([val_chemical_seq_vectors, val_chemical_embs,
                          val_protein_seq_vectors, val_protein_embs],
                         np.array(val_labels))

        # train_stats_over_epochs = {'mse': [], 'rmse': [], 'r2': []}
        # val_stats_over_epochs = train_stats_over_epochs.copy()
        self.model.fit(x=[train_chemical_seq_vectors, train_chemical_embs,
                          train_protein_seq_vectors, train_protein_embs],
                       y=train_labels,
                       validation_data=val_tuple,
                       batch_size=self.batch_size,
                       epochs=self.n_epochs)
        '''
        train_stats = evaluate_predictions(y_true=train_labels,
                                           y_preds=self.predict(
                                               train_chemicals, train_chemical_embs,
                                               train_proteins, train_protein_embs),
                                           metrics=list(train_stats_over_epochs.keys()))
        for metric, stat in train_stats.items():
            train_stats_over_epochs[metric].append(stat)

        if val_tuple is not None:
            val_stats = evaluate_predictions(y_true=val_labels,
                                             y_preds=self.predict(
                                                 val_tuple[0], val_tuple[1]),
                                             metrics=list(val_stats_over_epochs.keys()))
            for metric, stat in val_stats.items():
                val_stats_over_epochs[metric].append(stat)

        # self.history['train'] = train_stats_over_epochs
        # if val_stats_over_epochs is not None:
            # self.history['val'] = val_stats_over_epochs
        '''
        self.history = self.model.history.history
        return self.model.history.history

    def predict(self, chemical_seqs, chemical_embs, protein_seqs, protein_embs):
        chemical_vectors = self.vectorize_chemicals(chemical_seqs)
        protein_vectors = self.vectorize_proteins(protein_seqs)
        return self.model.predict([chemical_vectors, chemical_embs, protein_vectors, protein_embs]).tolist()

    def save(self, path):
        # path = strip_path(path)
        # print('Saving the model')
        self.model.save(f'{path}/model')

        with open(f'{path}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

        donot_copy = {'model', 'history'}
        dct = {k: v for k, v in self.__dict__.items() if k not in donot_copy}
        with open(f'{path}/params.json', 'w') as f:
            json.dump(dct, f, indent=4)
