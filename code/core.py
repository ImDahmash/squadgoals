from functools import wraps


def abstractmethod(f):
    """
    Marks the provided method as abstract, raising an error if not overridden.
    """
    @wraps(f)
    def inner(*args, **kwargs):
        cname = args[0].__class__.__name__  # Grab value of `self` at time of call
        raise NotImplementedError("Method {}.{} not implemented!".format(cname, f.__name__))
    return inner


class Config(object):
    """
    Configuration file with sane defaults
    """

    def __init__(self, options):
        self._max_length = options.get("max_length", 100)
        self._keep_prob = options.get("keep_prob", 0.99)

    @property
    def max_length(self):
        return self._max_length

    @property
    def keep_prob(self):
        return self._keep_prob


class SquadModel(object):
    """
    Model for processing and answering the SQuAD dataset.

    Models receive sets of (question, context, answer) tuples during training,
    and are responsible for putting these into a format that they can understand. This can be by deriving
    BOW features and using a classical model e.g. Logistic Regression, or by using the IDs to form a set of
    word embedding features which can be processed by a fancy RNN-style algo.
    """

    @abstractmethod
    def initialize(self, config):
        """
        Initialize the model based off the given configuration
        :param config: An instance of Config with important information for configuring.
        :return: None
        """
        pass

    @abstractmethod
    def train_batch(self, question_ids, passage_ids, answer_ids, answer_start, answer_end):
        """
        Train on a set of data of a fixed size. All the parameters have the same length

        :param question_ids: A list of lists of token IDs representing the question for each batch member
        :param passage_ids: A list of lists of token IDs representing the passage for each batch member
        :param answer_ids: A list of lists of token IDs representing the vocabulary ID of each token in the answer
        :param answer_start: The index of the first token in the answer within the passage
        :param answer_end: The index of the last token of the answer within the passage
        :return: Nothing
        """
        pass

    @abstractmethod
    def predict(self, question_ids, passage_ids):
        """
        Ask the model to perform predictions about the given set of things
        :param question_ids: A list of lists of token IDs representing the question for each batch member
        :param passage_ids: A list of lists of token IDs representing the passage for each batch member
        :return: A list of the form [(answer_start, answer_end)] for each item in the batch indicating the answers
        """
        pass
