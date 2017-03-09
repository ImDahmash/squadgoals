class Config(object):
    """
    Configuration file with sane defaults
    """

    def __init__(self, options):
        self.max_length = options.get("max_length", 100)
        self.keep_prob = options.get("keep_prob", 0.99)
        self.num_classes = options.get("num_classes", 2)
        self.embedding_size = options.get("embedding_size", 100)
        self.hidden_size = options.get("hidden_size", 150)
        self.cell_type = options.get("cell_type", "lstm")


class SquadModel(object):
    """
    Model for processing and answering the SQuAD dataset.

    Models receive sets of (question, context, answer) tuples during training,
    and are responsible for putting these into a format that they can understand. This can be by deriving
    BOW features and using a classical model e.g. Logistic Regression, or by using the IDs to form a set of
    word embedding features which can be processed by a fancy RNN-style algo.
    """

    def initialize(self, config):
        """
        Initialize the model based off the given configuration
        :param config: An instance of Config with important information for configuring.
        :return: None
        """
        raise NotImplementedError("Abstract method")

    def train_batch(self, sess, question_ids, passage_ids, answer_start, answer_end):
        """
        Train on a set of data of a fixed size. All the parameters have the same length

        :param sess: tf.Session to run in
        :param question_ids: A list of lists of token IDs representing the question for each batch member
        :param passage_ids: A list of lists of token IDs representing the passage for each batch member
        :param answer_start: The index of the first token in the answer within the passage
        :param answer_end: The index of the last token of the answer within the passage
        :return: Nothing
        """
        raise NotImplementedError("Abstract method")

    def predict(self, question_ids, passage_ids):
        """
        Ask the model to perform predictions about the given set of things
        :param question_ids: A list of lists of token IDs representing the question for each batch member
        :param passage_ids: A list of lists of token IDs representing the passage for each batch member
        :return: A list of the form [(answer_start, answer_end)] for each item in the batch indicating the answers
        """
        raise NotImplementedError("Abstract method")

    def checkpoint(self, save_dir):
        """
        Save the current set of parameters to disk.
        :param save_dir: Path to the directory where saved parameters are written.
        :return: Nothing.
        """
        pass


class EncoderDecoderModel(object):
    """
    Model for objects which contain an explicit encode and decode step.
    """

    def encode(self, paragraph_batch, question_batch):
        """
        Encodes the paragraph and question batches into a set of hidden states.
        :param paragraph_batch: A Tensor of dimension [batch_size, None, embedding_size] holding paragraph wordvectors
        :param question_batch: A Tensor of dimension [batch_size, None, embedding_size] holding question wordvectors
        :return:
        """
        raise NotImplementedError("Abstract method")

    def decode(self, h_q, h_p, attention):
        """
        Decodes the hidden state as well as the attention vector into a prediction.
        :param h_q: Abstract representation of question (shape `[batch_size, max_length, 2*hidden_size]`)
        :param h_p: Abstract representation of paragraph (shape `[batch_size, max_length, 2*hidden_size]`)
        :param attention: Matrix representing a function of attention to the paragraph.
        :return: The predicted span (start_index, end_index)
        """
        raise NotImplementedError("Abstract method")
