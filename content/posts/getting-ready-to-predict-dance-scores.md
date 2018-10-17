---
title: "Getting Ready to Predict Dance Scores"
date: 2018-10-16T22:52:06-04:00
draft: false
---

So! After getting the data prepared in [the previous post](../data-prep), I should be ready to use machine learning and build models? Not quite. First, I need to think a little more about the requirements of the model, how I will use the available data, and how to prepare the data for the model.

## Model requirements

I want to see how well a model can predict the total score a partnership will receive for a given dance. The goal is to be able to predict the scores for an upcoming show ahead of time, and see how well the model performs!

I want to predict the total score, out of 40, rather than the individual judges' scores, because with for the total score I have the largest training set possible: all dances from all series of Strictly. (I was surprised just how many judges there have been! But also, [Len forever](https://giphy.com/gifs/bbc-xUOxfljQpQdYB4r9Yc/tile).) Of course, the total score may in part depend on the identity of the four judges.

I'll be using the python package `scikit-learn` (`sklearn`) to implement my models. The package conveniently has all kinds of different models available, along with plentiful libraries for pre-processing data, training models, evaluating models, and more.

## Model inputs

I brainstormed factors in the data set that may be important in predicting the scores:

- the identity of the celebrity
- the identity of the professional
- the type of dance
- how far along in the series the week is, fractionally
- the order of the dance in the show, fractionally (see below)
- which series it is (this should have baked in a lot of the dependence on the judging panel, assuming it's typically the same judging lineup for an entire series)

Of course, there are other factors that I could also imagine playing a role (e.g., celebrity demographic/biographical information), but I've limited myself to the above for now. Of course, it's also likely that a dance's score isn't entirely deterministic, so I wouldn't expect any model to be accurate one hundred percent of the time.

# Getting prepared data ready for modeling

For all models, I first need to define the inputs (factors used to make the predictions) and labels (what is being predicted), by choosing the appropriate columns from the `df_4judges` DataFrame constructed [last time](../data-prep):

```python
all_inputs = df_4judges[['celebrity','professional','dance','normed_week','series']]
all_labels = df_4judges['total']
```

You'll notice I left out the order of the dance in the show as an input. This is because I realized the order is not announced ahead of time! So, even if the dance order is a helpful predictive variable, it can't be used to predict scores for the upcoming week.

I also need to split the total data set into two parts: one part that is used to train each model, and one part that is used to test the trained model. Holding back part of the data for testing is key in checking for over-fitting to the training set. This is easy to set up in `sklearn`:

```python
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25, random_state=1)
```

And finally, I need to consider some of the inputs are numerical, whereas others should be treated categorically. Out of the box, many `sklearn` estimators don't support categorical inputs, but I found you can use a [`OneHotEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) (from `sklearn.preprocessing`) to treat these cases using [one-hot encoding](https://en.wikipedia.org/wiki/One-hot).

In order to only transform certain types of inputs, I used a ColumnTransformer (from `sklearn.compose`). This will serve as a pre-processor in constructing the models.

```python
num_columns = ['normed_week','series']
num_transformer = 'passthrough'

cat_columns = ['celebrity','professional','dance']
# explicitly include all categories since possibly will be only in training data and not testing
cat_categories = [list(all_inputs[x].unique()) for x in all_inputs[cat_columns]]
categorical_transformer = OneHotEncoder(categories=cat_categories)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', categorical_transformer, cat_columns)])
```

Now I'm ready to set up the models, which will happen next time! And remember, keeeeeeeeeeeeep data-ing!
