def prepare_dataset_for_training(nlp_dataset):
    """Changes an `nlp` dataset into the proper format for tokenization."""

    def prepare_example_dict(ex):
        """Returns the values in order corresponding to the data.
        ex:
            'Some text input'
        or in the case of multi-sequence inputs:
            ('The premise', 'the hypothesis',)
        etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))
    return list(text), list(outputs)
