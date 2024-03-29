### <h1>Week 2 (Fundamentals, ML Specific Concepts)</h1>

  <h2>"An Introduction to Machine Learning"</h2>

The field of study known as machine learning is concerned with the question of how to construct computer programs that automatically improvew with experience.
Examples
• A robot driving learning problem
• Handwriting recognition learning problem

A computer program which learns from experience is called a machine learning program or simply a learning program.
Classification of Machine Learning

<h3>
1.Supervised Learning
2.Unsupervised Learning
3.Reinforcement Learning
4.Semi-Supervised Learning
</h3>

<h2>"Supervised learning"</h2>
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.The given data is labeled.
Both classification and regression problems are supervised learning problems.

IMAGE

<h2>"Unsupervised learning"</h2>
Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses.
In unsupervised learning algorithms,classification or categorization is not included in the observations.

IMAGE

<h2>"Reinforcement learning"</h2>
Reinforcement learning is the problem of getting an agent to act in the world so as to maximize its rewards.
A learner is not told what actions to take as in most forms of machine learning but instead must discover which actions yield the most reward by trying them.

IMAGE

<h2>"Semi-supervised learning"</h2>
Semi-supervised learning is an approach to machine learning that combines small labeled data with a large amount of unlabeled data during training.
Semi-supervised learning falls between unsupervised learning and supervised learning

IMAGE

<h2>Machine Learning Model Development Steps</h2> 
<h3>
1.Collecting Data
2.Preparing the Data
3.Choosing a Model
4.Training the Model
5.Evaluating the Model
6.Parameter Tuning
7.Making Predictions
</h3>

IMAGE
IMAGE

<h2>"The `re` module"</h2>
•Data Preprocessing
•Feature Extraction
•Text Tokenization
•Text Matching and Filtering

<h2>"Pandas" is a popular open-source Python library</h2>
• Data Structures
• Data Loading and Saving
• Data Exploration and Manipulation
• Feature Engineering
• Data Transformation
• Integration with ML Libraries

<h2>"NumPy (Numerical Python)" is a fundamental package for numerical computing in Python.</h2>
• Multidimensional Arrays
• Mathematical Operations
• Linear Algebra
• Random Number Generation
• Array Manipulation
• Integration with Other Libraries

<h2>"Seaborn" is a Python data visualization library</h2>
• Statistical Visualization
• Pairwise Relationship
• Categorical Data Visualization
• Regression Analysis
• Distribution Visualization
• Customization and Styling

<h2>"Matplotlib" is a comprehensive library for creating static, animated, and interactive</h2>
visualizations in Python.
• Basic Plots
• Customization and Styling
• Subplots and Layouts
• Annotations and Text
• 3D Plotting
• Interactive Visualization
• Integration with Libraries
• Exporting and Saving Plots

<h2>"WordCloud" is a Python library used for generating word clouds, which are visual
representations of text data where the size of each word corresponds to its frequency or
importance within the text.</h2>
• Word Frequency Calculation
• Customization Options
• Visualization of Textual Data
• Text Preprocessing
• Topics Identification and Analysis
• Result Interpretation and Communication

<h2>The "train-test split" is a fundamental technique used in machine learning (ML) projects for
evaluating the performance of predictive models.</h2>
• Dataset Partitioning
• Evaluation of Model Performance
• Preventing Overfitting
• Parameter Tuning and Model Selection
• Cross Validation
• Stratified Splitting

<h2>A "confusion matrix" is a performance measurement tool used in machine learning (ML)
projects, particularly in classification tasks.</h2>
• Basic Structure
• Evaluation Metrics
• Visualization
• Error Analysis
• Model Comparison
• Threshold Optimization

<h2>A "classification report" is a summary of the performance of a classification model, typically
generated using the predictions made by the model and the actual labels from the test
dataset.</h2>
• Metrics Included
• Interpretation
• Visualization
• Model Comparison
• Error Analysis
• Threshold Optimization

<h2>"Stemming" is a text processing technique used in natural language processing (NLP) to
reduce words to their root or base form.</h2>
• Normalization of Text
• Feature Extraction
• Information Retrieval
• Text Preprocessing
• Implementation
• Limitations

<h2>"Lemmatization" is a technique used in natural language processing (NLP) and machine
learning (ML) projects to reduce words to their base or root form.</h2>
• POS Tagging
• Normalization
• Word Sense Disambiguation
• Implementation

<h2>A "regexp tokenizer", short for regular expression tokenizer, is a tokenizer used in natural
language processing (NLP) tasks to split text into individual tokens based on specific
patterns defined by regular expressions.</h2>
• Tokenization
• Regular Expressions
• Customization
• Handling Special Cases
• Language Specific Tokenization

<h2>The "Receiver Operating Characteristic (ROC)" curve and Area Under the Curve (AUC) are
evaluation metrics used in machine learning (ML) projects, particularly for binary
classification tasks.</h2>
• ROC Curve
• Interpretation
• AUC
• Evaluation
• Threshold Selection
• Implementation

<h2>"Natural Language Processing (NLP)" is a field of artificial intelligence (AI) and
computational linguistics focused on enabling computers to understand, interpret, and
generate human language in a meaningful way.</h2>
• NLP Tasks
• NLTK Library
• Functionalities of NLTK
• Integration with ML

<h2>"Logistic regression" is a statistical method used for binary classification tasks in machine
learning (ML) projects.</h2>
• Model Representation
• Decision Boundary
• Training
• Regularization
• Evaluation
• Implementation Example

<h2>"Bernoulli Naive Bayes" is a variant of the Naive Bayes algorithm, specifically designed for
binary classification tasks where the features are binary-valued (e.g., presence or absence
of a feature).</h2>
• Model Representation
• Parameter Estimation
• Prediction
• Smoothing
• Evaluation
• Implementation Example

<h2>"Support Vector Machine (SVM)" is a powerful supervised learning algorithm used for
classification, regression, and outlier detection tasks in machine learning (ML) projects.</h2>
• Objective
• Linear Seperability
• Kernel Trick
• Margin Maximization
• Regularization
• Dual Optimization Problem
• Implementation Example

<h2>"TF-IDF (Term Frequency-Inverse Document Frequency)" is a numerical statistic used in
natural language processing (NLP) and information retrieval to evaluate the importance of
a word in a document relative to a collection of documents.</h2>
• TF
• IDF
• TF – IDF Calculation
• Functionalities



<h2>"Bernoulli Naive Bayes"<h2> 

To understand Bernoulli Naive Bayes algorithm,
it is essential to understand Naive Bayes.
Naive Bayes is a supervised machine learning
algorithm to predict the probability of different
classes based on numerous attributes. It
indicates the likelihood of occurrence of an
event. Naive Bayes is also known as conditional
probability.
Naive Bayes is based on the Bayes Theorem.

IMAGE

The Naive Bayes classifier is based on two essential assumptions:-
(i) Conditional Independence - All features are independent of each other. This implies that
one feature does not affect the performance of the other. This is the sole reason behind the
‘Naive’ in ‘Naive Bayes.’
(ii) Feature Importance - All features are equally important. It is essential to know all the
features to make good predictions and get the most accurate results.

Let there be a random variable 'X' and let the probability of success be denoted by 'p' and
the likelihood of failure be represented by 'q.'
Success: p
Failure: q
q = 1 - (probability of Sucesss)
q = 1 - p

IMAGE

In the provided dataset, we are trying to predict whether a person has a disease or not
based on their age, gender, and fever. Here, ‘Disease’ is the target, and the rest are the features.

All values are binary.

We wish to classify an instance ‘X’ where Adult=’Yes’, Gender= ‘Male’, and Fever=’Yes’

Firstly, we calculate the class probability, probability of disease or not.
P(Disease = True) = ⅗
P(Disease = False) = ⅖
Secondly, we calculate the individual probabilities for each feature.
P(Adult= Yes | Disease = True) = ⅔
P(Gender= Male | Disease = True) = ⅔
P(Fever= Yes | Disease = True) = ⅔
P(Adult= Yes | Disease = False) = ½
P(Gender= Male | Disease = False) = ½
P(Fever = Yes | Disease = False) = ½

Now, we need to find out two probabilities:-
(i) P(Disease= True | X) = (P(X | Disease= True) _ P(Disease=True))/ P(X)
(ii) P( Disease = False | X) = (P(X | Disease = False) _ P(Disease= False) )/P(X)
P(Disease = True | X) = (( ⅔ _⅔ _ ⅔ ) _ (⅗))/P(X) = (8/27 _ ⅗) / P(X) = 0.17/P(X)
P(Disease = False | X) = [(½ * ½ * ½ ) * (⅖)] / P(X) = [⅛ * ⅖] / P(X) = 0.05/ P(X)

Now, we calculate estimator probability:-
P(X) = P(Adult= Yes) _ P(Gender = Male ) _ P(Fever = Yes)
= ⅗ _ ⅗ _ ⅗ = 27/125 = 0.21
So we get finally:- Now,we notice that (1) > (2),

                                                  the result of instance ‘X’ is
                                                  ‘True’, i.e., the person has
                                                  the disease.

P(Disease = True | X) = 0.17 / P(X)
= 0.17 / 0.21
= 0.80 - (1)
P(Disease = False | X) = 0.05 / P(X)
= 0.05 / 0.21
= 0.23 - (2)

