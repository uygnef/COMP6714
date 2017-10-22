## Submission.py for COMP6714-Project2
###################################################################################################################
import spacy
import zipfile
import pickle
import tensorflow as tf

def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    file = open('tokenize.pickle', 'rb')
    data = pickle.load(file)
    print(data[:100])



def process_data(input_data):
    """Extract the first file enclosed in a zip file as a list of words."""
    data = ''
    with zipfile.ZipFile(input_data) as f:
        for i in f.namelist():
            data += tf.compat.as_str(f.read(i))
    nlp = spacy.load('en')
    paredDate = nlp(data)
    for i in paredDate:
        print(i.lemma_, i.is_stop)
    ret = [tok.lemma_ for tok in paredDate if not tok.is_stop and tok.lemma_]
    for i in ret:
        print(i)
    file = open('tokenize.pickle', 'wb')
    pickle.dump(ret, file)
    return ret


def Compute_topk(model_file, input_adjective, top_k):
    pass # Remove this pass line, you need to implement your code to compute top_k words similar to input_adjective

if __name__ == "__main__":
    process_data('BBC_Data.zip')
    adjective_embeddings(1, 1, 1, 1)