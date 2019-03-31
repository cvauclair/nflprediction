import math
import functools
import torch
import torch.utils.data

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

# model: PyTorch model to be tested
# dataset: PyTorch DataSet
# k: k fold cross validation
# training_function: a function of the form foo(model, ) which returns the trained model
# testing_function: a function of the form foo(model, ) which returns the model's loss
def cross_validation(model, dataset, k, training_function, testing_function):
    if k <= 1:
        raise ValueError("k <= 1")

    # Get lenghts of subsets
    subset_size = math.floor(len(dataset)/k)
    # "k - 1" subsets of size "subset_size" and 1 subset containing the rest
    lengths = ([subset_size] * (k - 1)) + [len(dataset) - subset_size * (k - 1)]
    subsets = torch.utils.data.random_split(dataset, lengths)

    # For each subset, test on it and train on the rest (k-fold cross validation)
    losses = []
    for i in range(0, k):
        # Randomize model weigths
        model.apply(init_weights)

        training_dataset = functools.reduce((lambda x, y: x + y), [subsets[n] for n in range(0, k) if n != i])
        testing_dataset = subsets[i]

        # Training
        model = training_function(model, training_dataset)

        # Testing
        losses.append(testing_function(model, testing_dataset))

    # Return average loss
    return sum(losses)/len(losses)

def holdout_validation(model, dataset, testing_ratio, training_function, testing_function):
    if testing_ratio < 0 or testing_ratio > 1:
        raise ValueError("Invalid testing ratio")

    model.apply(init_weights)

    testing_set_size = math.floor(len(dataset) * testing_ratio)
    lengths = [len(dataset) - testing_set_size, testing_set_size]

    training_dataset, testing_dataset = torch.utils.data.random_split(dataset, lengths)

    # Training
    model = training_function(model=model, dataset=training_dataset)

    # Testing
    return testing_function(model=model, dataset=testing_dataset)