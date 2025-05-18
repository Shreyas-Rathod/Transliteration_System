import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense

def build_seq2seq_model(cell_type='LSTM', 
                        vocab_size=100, 
                        embedding_dim=64, 
                        hidden_size=128, 
                        num_encoder_layers=1, 
                        num_decoder_layers=1,
                        max_seq_len=20):

    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')
    embedded_encoder_input = embedding_layer(encoder_inputs)

    RNNCell = {'LSTM': LSTM, 'GRU': GRU, 'RNN': SimpleRNN}[cell_type]
    encoder_output = embedded_encoder_input
    encoder_states = []

    for i in range(num_encoder_layers):
        return_sequences = False if i == num_encoder_layers - 1 else True
        rnn_layer = RNNCell(hidden_size, return_sequences=return_sequences, return_state=True, name=f'encoder_rnn_{i+1}')
        if cell_type == 'LSTM':
            encoder_output, state_h, state_c = rnn_layer(encoder_output)
            encoder_states = [state_h, state_c]
        else:
            encoder_output, state = rnn_layer(encoder_output)
            encoder_states = [state]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    embedded_decoder_input = embedding_layer(decoder_inputs)

    decoder_output = embedded_decoder_input
    decoder_states_inputs = encoder_states
    decoder_states = []

    for i in range(num_decoder_layers):
        return_sequences = True
        rnn_layer = RNNCell(hidden_size, return_sequences=return_sequences, return_state=True, name=f'decoder_rnn_{i+1}')
        if cell_type == 'LSTM':
            decoder_output, _, _ = rnn_layer(decoder_output, initial_state=decoder_states_inputs)
        else:
            decoder_output, _ = rnn_layer(decoder_output, initial_state=decoder_states_inputs)

    decoder_dense = Dense(vocab_size, activation='softmax', name='output_dense')
    decoder_outputs = decoder_dense(decoder_output)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
