import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from PIL import Image
import random
import string
import io
import wandb

wandb.init(project="DA6401", name="lstm_attention_visualization")

def generate_dataset(size=1000, min_length=5, max_length=15):
    data = []
    for _ in range(size):
        length = random.randint(min_length, max_length)
        s = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
        data.append(s)
    return data

data = generate_dataset()

all_chars = sorted(list(set(''.join(data))))
char_to_idx = {c: i for i, c in enumerate(all_chars)}
idx_to_char = {i: c for i, c in enumerate(all_chars)}
vocab_size = len(all_chars)

X = []
y = []
max_len = 10

for text in data:
    for i in range(0, len(text) - max_len):
        X.append([char_to_idx[c] for c in text[i:i+max_len]])
        y.append(char_to_idx[text[i+max_len]])

X = np.array(X)
y = np.array(y)

y_onehot = np.zeros((len(y), vocab_size))
for i, label in enumerate(y):
    y_onehot[i, label] = 1

X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, units, return_sequences=False, **kwargs):
        super(CustomLSTM, self).__init__(units, return_sequences=return_sequences, **kwargs)
        self.activations = None
        self.inputs = None
        
    def call(self, inputs, **kwargs):
        self.inputs = inputs
        outputs = super(CustomLSTM, self).call(inputs, **kwargs)
        return outputs
        
    def get_config(self):
        config = super(CustomLSTM, self).get_config()
        return config

embedding_dim = 32
lstm_units = 64

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    CustomLSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.build((None, max_len))
print(model.summary())

lstm_layer = model.layers[1]

batch_size = 128
epochs = 3

class ActivationLogger(tf.keras.callbacks.Callback):
    def __init__(self, input_model, lstm_layer, val_data, idx_to_char, log_freq=1):
        super(ActivationLogger, self).__init__()
        self._model = input_model
        self.lstm_layer = lstm_layer
        self.val_data = val_data
        self.idx_to_char = idx_to_char
        self.log_freq = log_freq
        
        self.activation_model = None
        
    def on_train_begin(self, logs=None):
        self.activation_model = Model(
            inputs=self._model.input,
            outputs=[self._model.layers[0].output, self.lstm_layer.output]
        )
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq == 0:
            self.log_activations(epoch)
            
    def log_activations(self, epoch):
        sample_idx = np.random.randint(0, len(self.val_data[0]))
        input_seq = self.val_data[0][sample_idx:sample_idx+1]
        
        embeddings, lstm_output = self.activation_model.predict(input_seq)
        
        attention_map = self.create_attention_visualization(input_seq[0], lstm_output[0])
        
        wandb.log({
            f"attention_map_epoch_{epoch}": wandb.Image(attention_map),
            "epoch": epoch
        })
        
    def create_attention_visualization(self, input_seq, lstm_output):
        input_chars = [self.idx_to_char[idx] for idx in input_seq]
        
        input_tensor = np.array([input_seq])
        predictions = self._model.predict(input_tensor)[0]
        predicted_chars = [self.idx_to_char[np.argmax(pred)] for pred in [predictions]]
        
        emb_model = Model(inputs=self._model.input, outputs=self._model.layers[0].output)
        embeddings = emb_model.predict(input_tensor)[0]
        
        correlation_matrix = np.zeros((1, len(input_chars)))
        
        lstm_output_flat = lstm_output.reshape(1, -1)
        for i in range(len(input_chars)):
            emb_flat = embeddings[i].reshape(1, -1)
            similarity = np.dot(lstm_output_flat, emb_flat.T) / (
                np.linalg.norm(lstm_output_flat) * np.linalg.norm(emb_flat)
            )
            correlation_matrix[0, i] = similarity
            
        plt.figure(figsize=(10, 4))
        plt.imshow(correlation_matrix, cmap='viridis')
        plt.xticks(range(len(input_chars)), input_chars)
        plt.xlabel('Input Characters')
        plt.yticks([0], [f"Predicted: '{predicted_chars[0]}'"])
        plt.ylabel('Output Character')
        plt.title('Character-Level Attention Map')
        plt.colorbar(label='Attention Weight')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)

activation_logger = ActivationLogger(
    input_model=model,
    lstm_layer=lstm_layer,
    val_data=(X_val, y_val),
    idx_to_char=idx_to_char,
    log_freq=1
)

wandb_callback = wandb.keras.WandbCallback()

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[activation_logger, wandb_callback]
)

def create_detailed_attention_visualization(model, input_text, char_to_idx, idx_to_char):
    input_seq = [char_to_idx.get(c, 0) for c in input_text]
    if len(input_seq) > max_len:
        input_seq = input_seq[:max_len]
    elif len(input_seq) < max_len:
        input_seq = [0] * (max_len - len(input_seq)) + input_seq
        
    emb_model = Model(inputs=model.input, outputs=model.layers[0].output)
    
    lstm_layer = model.layers[1]
    
    input_tensor = np.array([input_seq])
    embeddings = emb_model.predict(input_tensor)[0]
    
    output_text = ""
    current_input = input_tensor.copy()
    
    attention_matrix = np.zeros((len(input_text), max_len))
    
    for i in range(max_len):
        pred = model.predict(current_input)[0]
        next_char_idx = np.argmax(pred)
        next_char = idx_to_char[next_char_idx]
        output_text += next_char
        
        lstm_output_model = Model(inputs=model.input, outputs=lstm_layer.output)
        lstm_output = lstm_output_model.predict(current_input)[0]
        
        for j in range(len(input_text)):
            lstm_output_flat = lstm_output.reshape(1, -1)
            emb_flat = embeddings[j].reshape(1, -1)
            similarity = np.dot(lstm_output_flat, emb_flat.T) / (
                np.linalg.norm(lstm_output_flat) * np.linalg.norm(emb_flat)
            )
            attention_matrix[j, i] = similarity[0, 0]
            
        new_seq = list(current_input[0])
        new_seq = new_seq[1:] + [next_char_idx]
        current_input[0] = np.array(new_seq)
        
    plt.figure(figsize=(12, 10))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.xticks(range(len(output_text)), list(output_text))
    plt.xlabel('Output Characters (Predicted)')
    plt.yticks(range(len(input_text)), list(input_text))
    plt.ylabel('Input Characters')
    plt.title('Character-Level Attention Map (Connectivity Visualization)')
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf), output_text

sample_input = "hello world"
final_vis, output_text = create_detailed_attention_visualization(
    model, sample_input, char_to_idx, idx_to_char
)

wandb.log({
    "final_attention_visualization": wandb.Image(final_vis),
    "input_text": sample_input,
    "predicted_text": output_text
})

model.save("lstm_attention_model.h5")
wandb.save("lstm_attention_model.h5")

wandb.log({
    "model_summary": str(model.summary()),
    "epochs": epochs,
    "batch_size": batch_size,
    "embedding_dim": embedding_dim,
    "lstm_units": lstm_units,
    "vocabulary_size": vocab_size,
    "max_sequence_length": max_len
})

print("Training complete. Visualizations available in W&B.")
print(f"W&B project: {wandb.run.project}")
print(f"W&B run: {wandb.run.name}")

wandb.finish()
