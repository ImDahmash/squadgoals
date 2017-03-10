import tensorflow as tf


class Config(object):
    """
    Configuration file with sane defaults
    """

    def __init__(self, options):
        self.batch_size = options.get("batch_size", 3)
        self.max_length = options.get("max_length", 100)
        self.keep_prob = options.get("keep_prob", 0.99)
        self.num_classes = options.get("num_classes", 2)
        self.embed_size = options.get("embed_size", 100)
        self.hidden_size = options.get("hidden_size", 150)
        self.cell_type = options.get("cell_type", "lstm")
        self.save_dir = options.get("save_dir", "save")


class SquadModel(object):
    """
    Model for processing and answering the SQuAD dataset.

    Models receive sets of (question, context, answer) tuples during training,
    and are responsible for putting these into a format that they can understand. This can be by deriving
    BOW features and using a classical model e.g. Logistic Regression, or by using the IDs to form a set of
    word embedding features which can be processed by a fancy RNN-style algo.
    """

    def initialize_graph(self, config):
        """
        Initialize the model based off the given configuration
        :param config: An instance of Config with important information for configuring.
        :return: None
        """
        raise NotImplementedError("Abstract method")

    def train_batch(self, question_batch, passage_batch, answer_batch, sess=None):
        """
        Train on a set of data of a fixed size. All the parameters have the same first dimension (batch_size),
        passage_ids and answer_batch have same 2nd dimension as well.
        :param question_batch: Batch of questions
        :param passage_batch: Batch of contexts
        :param answer_batch: Batch of answers with the same first two shape components as passage_batch
        :param sess: Optional tf.Session for evaluation
        """
        raise NotImplementedError("Abstract method")

    def predict(self, question_batch, passage_batch, answer_batch, sess=None):
        """
        Ask the model to perform predictions about the given set of things
        :param question_batch: A list of lists of token IDs representing the question for each batch member
        :param passage_batch: A list of lists of token IDs representing the passage for each batch member
        :param answer_batch: Batch of answers with the same first two shape components as passage_batch
        :param sess: Optional tf.Session used for evaluation
        :return: A list of the form [(answer_start, answer_end)] for each item in the batch indicating the answers
        """
        raise NotImplementedError("Abstract method")

    def checkpoint(self, save_dir, sess=None):
        """
        Save the current set of parameters to disk.
        :param save_dir: Path to the directory where saved parameters are written
        :param sess: Optional tf.Session used for evaluation
        :return: Nothing.
        """
        raise NotImplementedError("Abstract method")

    def restore_from_checkpoint(self, save_dir, sess=None):
        """
        Restore the graph from the latest checkpoint in the given save_dir
        :param save_dir: Location where checkpoint files are saved
        :param sess: Optional tf.Session used for evaluation
        :return: Nothing.
        """
        raise NotImplementedError("Abstract method")
