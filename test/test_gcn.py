"""
This example implements the experiments on citation networks from the paper:
Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
Thomas N. Kipf, Max Welling
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GraphConv

# Load data
dataset = 'cora'
A, X, y, train_mask, val_mask, test_mask = citation.load_data(dataset)

# Parameters
channels = 16           # Number of channels in the first layer
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original size of node features
n_classes = y.shape[1]  # Number of classes
dropout = 0.5           # Dropout rate for the features
l2_reg = 5e-4 / 2       # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 200            # Number of training epochs
es_patience = 10        # Patience for early stopping

# Preprocessing operations
fltr = GraphConv.preprocess(A).astype('f4')
X = X.toarray()

# Model definition
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dp1 = Dropout(dropout)
        self.gcn1 = GraphConv(channels,
                         activation='relu',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False)
        self.dp2 = Dropout(dropout)
        self.gcn2 = GraphConv(n_classes,
                         activation='softmax',
                         use_bias=False)
    
    def call(self, graph):
        X_in = graph[0]
        A_in = graph[1]
        X = self.dp1(X_in)
        X = self.gcn1([X, A_in])
        X = self.dp2(X)
        X = self.gcn2([X, A_in])
        return X

X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

model = MyModel()

# output = model([X, fltr])

# print(output)

# Build model
# model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])


# Train model
validation_data = ([X, fltr], y, val_mask)
model.fit([X, fltr],
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X, fltr],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

model.summary()