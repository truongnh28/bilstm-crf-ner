from gensim.scripts.glove2word2vec import glove2word2vec

def run():
    glove_input_file = './pre-trained/glove.42B.300d.txt'
    word2vec_output_file = './pre-trained/glove.42B.300d_word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)