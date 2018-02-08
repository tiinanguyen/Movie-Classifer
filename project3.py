
# coding: utf-8

# # Project 3 - Classification
# Welcome to the third project of Data 8!  You will build a classifier that guesses whether a movie is romance or action, using only the numbers of times words appear in the movies's screenplay.  By the end of the project, you should know how to:
# 
# 1. Build a k-nearest-neighbors classifier.
# 2. Test a classifier on data.
# 
# ### Logistics
# 
# 
# **Deadline.** This project is due at 11:59pm on Thursday 11/30. You can earn an early submission bonus point by submitting your completed project by Wednesday 11/29. It's **much** better to be early than late, so start working now.
# 
# **Checkpoint.** For full credit, you must also **complete Part 1 of the project (out of 4) and submit them by 11:59pm on Tuesday 11/21**. You will have some lab time to work on these questions, but we recommend that you start the project before lab and leave time to finish the checkpoint afterward.
# 
# **Partners.** You may work with one other partner. It's best to work with someone in your lab. Only one of you is required to submit the project. On [okpy.org](http://okpy.org), the person who submits should also designate their partner so that both of you receive credit.
# 
# **Rules.** Don't share your code with anybody but your partner. You are welcome to discuss questions with other students, but don't share the answers. The experience of solving the problems in this project will prepare you for exams (and life). If someone asks you for the answer, resist! Instead, you can demonstrate how you would solve a similar problem.
# 
# **Support.** You are not alone! Come to office hours, post on Piazza, and talk to your classmates. If you want to ask about the details of your solution to a problem, make a private Piazza post and the staff will respond. If you're ever feeling overwhelmed or don't know how to make progress, email your TA or tutor for help. You can find contact information for the staff on the [course website](http://data8.org/fa17/staff.html).
# 
# **Tests.** Passing the tests for a question **does not** mean that you answered the question correctly. Tests usually only check that your table has the correct column labels. However, more tests will be applied to verify the correctness of your submission in order to assign your final score, so be careful and check your work!
# 
# **Advice.** Develop your answers incrementally. To perform a complicated table manipulation, break it up into steps, perform each step on a different line, give a new name to each result, and check that each intermediate result is what you expect. You can add any additional names or functions you want to the provided cells. 
# 
# To get started, load `datascience`, `numpy`, `plots`, and `ok`.

# In[1]:


# Run this cell to set up the notebook, but please don't change it.

import numpy as np
import math
from datascience import *

# These lines set up the plotting functionality and formatting.
import matplotlib
matplotlib.use('Agg', warn=False)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# These lines load the tests.
from client.api.notebook import Notebook
ok = Notebook('project3.ok')
_ = ok.auth(inline=True)


# # 1. The Dataset
# 
# In this project, we are exploring movie screenplays. We'll be trying to predict each movie's genre from the text of its screenplay. In particular, we have compiled a list of 5,000 words that might occur in the dialog of a movie. For each movie, our dataset tells us the frequency with which each of these words occurs in its screenplay. All words have been converted to lowercase.
# 
# Run the cell below to read the `movies` table. **It may take up to a minute to load.**

# In[2]:


movies = Table.read_table('movies.csv')
movies.where("Title", "the matrix").select(0, 1, 2, 3, 4, 5, 10, 30, 5005)


# The above cell prints a few columns of the row for the action movie *The Matrix*.  The movie contains 3792 words. The word "it" appears 115 times, as it makes up  $\frac{115}{3792} \approx 0.030327$ of the words in the movie. The word "not" appears 33 times, as it makes up $\frac{33}{3792} \approx 0.00870253$ of the words. The word "fling" doesn't appear at all.
# 
# This numerical representation of a body of text, one that describes only the frequencies of individual words, is called a bag-of-words representation. A lot of information is discarded in this representation: the order of the words, the context of each word, who said what, the cast of characters and actors, etc. However, a bag-of-words representation is often used for machine learning applications as a reasonable starting point, because a great deal of information is also retained and expressed in a convenient and compact format. In this project, we will investigate whether this representation is sufficient to build an accurate genre classifier.

# All movie titles are unique. The `row_for_title` function provides fast access to the one row for each title. 

# In[3]:


title_index = movies.index_by('Title')
def row_for_title(title):
    """Return the row for a title, similar to the following expression (but faster)
    
    movies.where('Title', title).row(0)
    """
    return title_index.get(title)[0]


# For example, the fastest way to find the frequency of "hey" in the movie *The Terminator* is to access the `'hey'` item from its row. Check the original table to see if this worked for you!

# In[4]:


row_for_title('the terminator').item('hey') 


# #### Question 1.1
# Set `expected_row_sum` to the number that you __expect__ will result from summing all proportions in each row, excluding the first six columns.

# In[5]:


# Set row_sum to a number that's the (approximate) sum of each row of word proportions.
expected_row_sum = 1


# In[6]:


_ = ok.grade("q1_1")
_ = ok.backup()


# Run the cell below to generate a histogram of the actual row sums. It should confirm your answer above, perhaps with a small amount of error.

# In[6]:


# Histogram of the sum of word frequency proportions for each movie.
Table().with_column('Sum', movies.drop(0, 1, 2, 3, 4, 5).apply(sum)).hist()


# This dataset was extracted from [a Kaggle dataset from Cornell University](https://www.kaggle.com/Cornell-University/movie-dialog-corpus). After transforming the dataset (e.g., converting the words to lowercase, removing the naughty words, and converting the counts to frequencies), we created this new dataset containing the frequency of 5000 common words in each movie.

# In[7]:


print('Words with frequencies:', movies.drop(np.arange(6)).num_columns) 
print('Movies with genres:', movies.num_rows)


# ## 1.1. Word Stemming
# The columns other than "Title", "Genre", "Year", "Rating", "# Votes" and "# Words" in the `movies` table are all words that appear in some of the movies in our dataset.  These words have been *stemmed*, or abbreviated heuristically, in an attempt to make different [inflected](https://en.wikipedia.org/wiki/Inflection) forms of the same base word into the same string.  For example, the column "manag" is the sum of proportions of the words "manage", "manager", "managed", and "managerial" (and perhaps others) in each movie.  
# 
# Stemming makes it a little tricky to search for the words you want to use, so we have provided another table that will let you see examples of unstemmed versions of each stemmed word.  Run the code below to load it.

# In[8]:


# Just run this cell.
vocab_mapping = Table.read_table('stem.csv')
stemmed = np.take(movies.labels, np.arange(3, len(movies.labels)))
vocab_table = Table().with_column('Stem', stemmed).join('Stem', vocab_mapping)
vocab_table.take(np.arange(1100, 1110))


# #### Question 1.1.1
# Assign `percent_unchanged` to the **percentage** of words in `vocab_table` that are the same as their stemmed form (such as "blame" above).
# 
# *Hint:* Try using `where` and comparing the number of rows in a table of only unchanged vocabulary with the number of rows in `vocab_table`.

# In[9]:


vocab_table


# In[10]:


num_unchanged = vocab_table.where(vocab_table.column('Stem') == vocab_table.column('Word')).num_rows
percent_unchanged = (num_unchanged / vocab_table.num_rows) * 100
print(round(percent_unchanged, 2), '% are unchanged')


# In[50]:


_ = ok.grade("q1_1_1")
_ = ok.backup()


# #### Question 1.1.2
# Assign `stemmed_message` to the stemmed version of the word "alternating".

# In[11]:


# Set stemmed_message to the stemmed version of "alternating" (which
# should be a string).  Use vocab_table.
stemmed_message = vocab_table.where('Word',are.equal_to('alternating')).column('Stem').item(0)
stemmed_message


# In[61]:


_ = ok.grade("q1_1_2")
_ = ok.backup()


# #### Question 1.1.3
# Assign `unstemmed_run` to an array of words in `vocab_table` that have "run" as its stemmed form. 

# In[12]:


# Set unstemmed_run to the unstemmed versions of "run" (which
# should be an array of string).
unstemmed_run = vocab_table.where('Stem', are.equal_to('run')).column('Word')
unstemmed_run


# In[71]:


_ = ok.grade("q1_1_3")
_ = ok.backup()


# #### Question 1.1.4
# What word in `vocab_table` that starts with an 'n' was shortened the most by this stemming process? Assign `most_shortened` to the word. It's an example of how heuristic stemming can collapse two unrelated words into the same stem (which is bad, but happens a lot in practice anyway).

# How do you find the number of characters in a word? How can you use this function on each item of a column? Two different questions here. Find difference between the length of stem & the word, make new column

# In[13]:


# In our solution, we found it useful to first make an array
# called shortened containing the number of characters that was
# chopped off of each word in vocab_table, but you don't have
# to do that.
length_table = vocab_table.with_column(
    'Length', vocab_table.apply(len, 'Word') - vocab_table.apply(len, 'Stem')
)
length_table

shortened = length_table.column('Length')
most_shortened = length_table.where('Length', are.equal_to(max(length_table.column('Length')))).column('Word').item(0)

# This will display your answer and its shortened form.
vocab_table.where('Word', most_shortened)


# In[98]:


_ = ok.grade("q1_1_4")
_ = ok.backup()


# ## 1.2. Splitting the dataset
# We're going to use our `movies` dataset for two purposes.
# 
# 1. First, we want to *train* movie genre classifiers.
# 2. Second, we want to *test* the performance of our classifiers.
# 
# Hence, we need two different datasets: *training* and *test*.
# 
# The purpose of a classifier is to classify unseen data that is similar to the training data. Therefore, we must ensure that there are no movies that appear in both sets. We do so by splitting the dataset randomly. The dataset has already been permuted randomly, so it's easy to split.  We just take the top for training and the rest for test. 
# 
# Run the code below (without changing it) to separate the datasets into two tables.

# In[14]:


# Here we have defined the proportion of our data
# that we want to designate for training as 17/20ths
# of our total dataset.  3/20ths of the data is
# reserved for testing.

training_proportion = 17/20

num_movies = movies.num_rows
num_train = int(num_movies * training_proportion)
num_valid = num_movies - num_train

train_movies = movies.take(np.arange(num_train))
test_movies = movies.take(np.arange(num_train, num_movies))

print("Training: ",   train_movies.num_rows, ";",
      "Test: ",       test_movies.num_rows)


# #### Question 1.2.1
# Draw a horizontal bar chart with two bars that show the proportion of Romance movies in each dataset.  Complete the function `romance_proportion` first; it should help you create the bar chart.

# In[15]:


movies.where('Genre', 'romance')


# In[16]:


def romance_proportion(table):
    """Return the proportion of movies in a table that have the Romance genre."""
    return table.where('Genre', 'romance').num_rows / table.num_rows

# The staff solution took 4 lines.  Start by creating a table.
romance = Table().with_column(
    'Name', make_array('Training', 'Testing'),
    'Proportions', make_array(romance_proportion(train_movies), romance_proportion(test_movies))
)
romance.barh(0,1)


# ### Checkpoint Reached
# 
# You have reached the project checkpoint. Please submit now in order to record your progress. If you go back and revise your answers in the section above after the checkpoint is due, that's fine. Your revised answers will be graded. However, you will only get credit for your checkpoint submission if you have passed the tests provided for every question above.
# 
# If you are working with a partner, only one of you needs to submit. For both of you to receive credit, the person who submits must invite the other to be their partner on [okpy.org](http://okpy.org). Please invite your partner now and tell them to accept the invitation **before** the checkpoint deadline!

# In[ ]:


_ = ok.submit()


# # 2. K-Nearest Neighbors - A Guided Example
# 
# K-Nearest Neighbors (k-NN) is a classification algorithm.  Given some *attributes* (also called *features*) of an unseen example, it decides whether that example belongs to one or the other of two categories based on its similarity to previously seen examples. Predicting the category of an example is called *labeling*, and the predicted category is also called a *label*.
# 
# An attribute (feature) we have about each movie is *the proportion of times a particular word appears in the movies*, and the labels are two movie genres: romance and action.  The algorithm requires many previously seen examples for which both the attributes and labels are known: that's the `train_movies` table.
# 
# To build understanding, we're going to visualize the algorithm instead of just describing it.

# ## 2.1. Classifying a movie
# 
# In k-NN, we classify a movie by finding the `k` movies in the *training set* that are most similar according to the features we choose. We call those movies with similar features the *nearest neighbors*.  The k-NN algorithm assigns the movie to the most common category among its `k` nearest neighbors.
# 
# Let's limit ourselves to just 2 features for now, so we can plot each movie.  The features we will use are the proportions of the words "money" and "feel" in the movie.  Taking the movie "Batman Returns" (in the test set), 0.000502 of its words are "money" and 0.004016 are "feel". This movie appears in the test set, so let's imagine that we don't yet know its genre.
# 
# First, we need to make our notion of similarity more precise.  We will say that the *distance* between two movies is the straight-line distance between them when we plot their features in a scatter diagram. This distance is called the Euclidean ("yoo-KLID-ee-un") distance, whose formula is $\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$.
# 
# For example, in the movie *Titanic* (in the training set), 0.0009768 of all the words in the movie are "money" and 0.0017094 are "feel".  Its distance from *Batman Returns* on this 2-word feature set is $\sqrt{(0.000502 - 0.0009768)^2 + (0.004016 - 0.0017094)^2} \approx 0.00235496$.  (If we included more or different features, the distance could be different.)
# 
# A third movie, *The Avengers* (in the training set), is 0 "money" and 0.001115 "feel".
# 
# The function below creates a plot to display the "money" and "feel" features of a test movie and some training movies. As you can see in the result, *Batman Returns* is more similar to *Titanic* than to *The Avengers* based on these features. However, we know that *Batman Returns* and *The Avengers* are both action movies, so intuitively we'd expect them to be more similar. Unfortunately, that isn't always the case. We'll discuss this more later.

# In[17]:


# Just run this cell.
def plot_with_two_features(test_movie, training_movies, x_feature, y_feature):
    """Plot a test movie and training movies using two features."""
    test_row = row_for_title(test_movie)
    distances = Table().with_columns(
            x_feature, [test_row.item(x_feature)],
            y_feature, [test_row.item(y_feature)],
            'Color',   ['unknown'],
            'Title',   [test_movie]
        )
    for movie in training_movies:
        row = row_for_title(movie)
        distances.append([row.item(x_feature), row.item(y_feature), row.item('Genre'), movie])
    distances.scatter(x_feature, y_feature, colors='Color', labels='Title', s=30)
    
training = ["titanic", "the avengers"] 
plot_with_two_features("batman returns", training, "money", "feel")
plots.axis([-0.001, 0.0015, -0.001, 0.006])


# #### Question 2.1.1
# 
# Compute the distance between the two action movies, *Batman Returns* and *The Avengers*, using the `money` and `feel` features only.  Assign it the name `romance_distance`. (Edit: It would make more sense conceptually to call this 'action_distance', unfortunately, the test looks specifically for the name 'romance_distance'.)
# 
# **Note:** If you have a row object, you can use `item` to get a value from a column by its name.  For example, if `r` is a row, then `r.item("Genre")` is the value in column `"Genre"` in row `r`.

# In[18]:


batman = row_for_title("batman returns") 
avengers = row_for_title("the avengers") 
romance_distance = ((batman.item('money') - avengers.item('money'))**2 + (batman.item('feel') - avengers.item('feel'))**2)**(1/2)
romance_distance


# In[22]:


_ = ok.grade("q2_1_1")
_ = ok.backup()


# Below, we've added a third training movie, *The Terminator*. Before, the point closest to *Batman Returns* was *Titanic*, a romance movie. However, now the closest point is *The Terminator*, an action movie.

# In[19]:


training = ["the avengers", "titanic", "the terminator"] 
plot_with_two_features("batman returns", training, "money", "feel") 
plots.axis([-0.001, 0.0015, -0.001, 0.006])


# #### Question 2.1.2
# Complete the function `distance_two_features` that computes the Euclidean distance between any two movies, using two features. The last two lines call your function to show that *Batman Returns* is closer to *The Terminator* than *The Avengers*. 

# In[20]:


def distance_two_features(title0, title1, x_feature, y_feature):
    """Compute the distance between two movies with titles title0 and title1
    
    Only the features named x_feature and y_feature are used when computing the distance.
    """
    row0 = row_for_title(title0)
    row1 = row_for_title(title1)
    distance = ((row0.item(x_feature) - row1.item(x_feature))**2 + ((row0.item(y_feature) - row1.item(y_feature)))**2)**(1/2)
    return distance

for movie in make_array("the terminator", "the avengers"):
    movie_distance = distance_two_features(movie, "batman returns", "money", "feel")
    print(movie, 'distance:\t', movie_distance)


# In[24]:


_ = ok.grade("q2_1_2")
_ = ok.backup()


# #### Question 2.1.3
# Define the function `distance_from_batman_returns` so that it works as described in its documentation. [Warning: It is important to make all movie titles lowercase, otherwise you'll get a confusing NoneType error.]

# In[21]:


def distance_from_batman_returns(title):
    """The distance between the given movie and "batman returns", based on the features "money" and "feel".
    
    This function takes a single argument:
      title: A string, the name of a movie.
    """
    batman = row_for_title('batman returns')
    other_movie = row_for_title(title)
    distance = ((batman.item('money') - other_movie.item('money'))**2 + (batman.item('feel') - other_movie.item('feel'))**2)**(1/2)
    return distance


# In[26]:


_ = ok.grade("q2_1_3")
_ = ok.backup()


# #### Question 2.1.4
# 
# Using the features `"money" and "feel"`, what are the names and genres of the 7 movies in the **training set** closest to "batman returns"?  To answer this question, make a table named `close_movies` containing those 7 movies with columns `"Title"`, `"Genre"`, `"money"`, and `"feel"`, as well as a column called `"distance"` that contains the distance from "batman returns".  The table should be **sorted in ascending order by `distance`**.

# In[22]:


# The staff solution took 4 lines.
    
results = Table().with_column(
    'Title', train_movies.column('Title'), 
    'Genre', train_movies.column('Genre'),
    'money', train_movies.column('money'),
    'feel', train_movies.column('feel'),
    'distance', train_movies.apply(distance_from_batman_returns, 'Title')
)
  
close_movies = results.sort('distance', descending = False).take(np.arange(0,7))
close_movies


# In[137]:


_ = ok.grade("q2_1_4")
_ = ok.backup()


# #### Question 2.1.5
# Define the function `most_common` so that it works as described in its documentation below.

# In[24]:


def most_common(label, table):
    """The most common element in a column of a table.
    
    This function takes two arguments:
      label: The label of a column, a string.
      table: A table.
     
    It returns the most common value in that column of that table.
    In case of a tie, it returns any one of the most common values
    """
    common = max(table.group(label).column('count'))
    title = table.group(label).where('count', common).column(label).item(0)
    return title

# Calling most_common on your table of 7 nearest neighbors classifies
# "batman returns" as a romance movie, 5 votes to 2. 
most_common('Genre', close_movies)


# In[150]:


_ = ok.grade("q2_1_5")
_ = ok.backup()


# Congratulations are in order -- you've classified your first movie! However, we can see that the classifier doesn't work too well since it categorized Batman Returns as a romance movie (unless you count the bromance between Alfred and Batman). Let's see if we can do better!

# # 3. Features

# Now, we're going to extend our classifier to consider more than two features at a time.
# 
# Euclidean distance still makes sense with more than two features. For `n` different features, we compute the difference between corresponding feature values for two movies, square each of the `n`  differences, sum up the resulting numbers, and take the square root of the sum.

# #### Question 3.1
# Write a function to compute the Euclidean distance between two **arrays** of features of *arbitrary* (but equal) length.  Use it to compute the distance between the first movie in the training set and the first movie in the test set, *using all of the features*.  (Remember that the first six columns of your tables are not features.)
# 
# **Note:** To convert row objects to arrays, use `np.array`. For example, if `t` was a table, `np.array(t.row(0))` converts row 0 of `t` into an array.

# In[28]:


def distance(features1, features2):
    """The Euclidean distance between two arrays of feature values."""
    return np.sqrt(sum((features1 - features2)**2))

first_training_set = np.array(train_movies.drop(np.arange(6)).row(0))
first_test_set = np.array(test_movies.drop(np.arange(6)).row(0))

distance_first_to_first = distance(first_training_set, first_test_set)
distance_first_to_first


# In[84]:


_ = ok.grade("q3_1")
_ = ok.backup()


# ## 3.1. Creating your own feature set
# 
# Unfortunately, using all of the features has some downsides.  One clear downside is *computational* -- computing Euclidean distances just takes a long time when we have lots of features.  You might have noticed that in the last question!
# 
# So we're going to select just 20.  We'd like to choose features that are very *discriminative*. That is, features which lead us to correctly classify as much of the test set as possible.  This process of choosing features that will make a classifier work well is sometimes called *feature selection*, or more broadly *feature engineering*.

# #### Question 3.0.1
# In this question, we will help you get started on selecting more effective features for distinguishing romance from action movies. The plot below (generated for you) shows the average number of times each word occurs in a romance movie on the horizontal axis and the average number of times it occurs in an action movie on the vertical axis. 

# ![alt text](word_plot.png "Title")

# The following questions ask you to interpret the plot above. For each question, select one of the following choices and assign its number to the provided name.
#     1. The word is uncommon in both action and romance movies
#     2. The word is common in action movies and uncommon in romance movies
#     3. The word is uncommon in action movies and common in romance movies
#     4. The word is common in both action and romance movies
#     5. It is not possible to say from the plot 

# What properties does a word in the bottom left corner of the plot have? Your answer should be a single integer from 1 to 5, corresponding to the correct statement from the choices above.

# In[32]:


bottom_left = 1


# In[86]:


_ = ok.grade("q3_0_1")
_ = ok.backup()


# What properties does a word in the bottom right corner have?

# In[33]:


bottom_right = 3


# In[88]:


_ = ok.grade("q3_0_2")
_ = ok.backup()


# What properties does a word in the top right corner have?

# In[34]:


top_right = 4


# In[90]:


_ = ok.grade("q3_0_3")
_ = ok.backup()


# What properties does a word in the top left corner have?

# In[35]:


top_left = 2


# In[92]:


_ = ok.grade("q3_0_4")
_ = ok.backup()


# If we see a movie with a lot of words that are common for action movies but uncommon for romance movies, what would be a reasonable guess about the genre of the movie? 
#     1. It is an action movie.
#     2. It is a romance movie.

# In[36]:


movie_genre_guess = 1


# In[94]:


_ = ok.grade("q3_0_5")
_ = ok.backup()


# #### Question 3.1.1
# Using the plot above, choose 20 common words that you think might let you distinguish between romance and action movies. Make sure to choose words that are frequent enough that every movie contains at least one of them. Don't just choose the 20 most frequent, though... you can do much better.
# 
# You might want to come back to this question later to improve your list, once you've seen how to evaluate your classifier.  

# In[37]:


# Set my_20_features to an array of 20 features (strings that are column labels)

my_20_features = make_array('afraid', 'marri', 'women', 'happi', 'command', 'captain', 'secur', 'record', 'protect', 'tonight', 'shot', 'save', 'trust', 'miss', 'husband', 'system', 'emerg', 'blood', 'war', 'secret')

train_20 = train_movies.select(my_20_features)
test_20 = test_movies.select(my_20_features)


# This test makes sure that you have chosen words such that at least one appears in each movie. If you can't find words that satisfy this test just through intuition, try writing code to print out the titles of movies that do not contain any words from your list, then look at the words they do contain.

# In[97]:


_ = ok.grade("q3_1_1")
_ = ok.backup()


# #### Question 3.1.2
# In two sentences or less, describe how you selected your features. 

# Since the 20 features are meant to distinguish between romance and action movies, I decided to pick several words on both sides of the spectrum with few in between that seem to lean more on side than the other. 

# Next, let's classify the first movie from our test set using these features.  You can examine the movie by running the cells below. Do you think it will be classified correctly?

# In[38]:


print("Movie:")
test_movies.take(0).select('Title', 'Genre').show()
print("Features:")
test_20.take(0).show()


# As before, we want to look for the movies in the training set that are most alike our test movie.  We will calculate the Euclidean distances from the test movie (using the 20 selected features) to all movies in the training set.  You could do this with a `for` loop, but to make it computationally faster, we have provided a function, `fast_distances`, to do this for you.  Read its documentation to make sure you understand what it does.  (You don't need to read the code in its body unless you want to.)

# In[54]:


# Just run this cell to define fast_distances.

def fast_distances(test_row, train_rows):
    """An array of the distances between test_row and each row in train_rows.

    Takes 2 arguments:
      test_row: A row of a table containing features of one
        test movie (e.g., test_20.row(0)).
      train_rows: A table of features (for example, the whole
        table train_20)."""
    assert train_rows.num_columns < 50, "Make sure you're not using all the features of the movies table."
    counts_matrix = np.asmatrix(train_rows.columns).transpose()
    diff = np.tile(np.array(test_row), [counts_matrix.shape[0], 1]) - counts_matrix
    distances = np.squeeze(np.asarray(np.sqrt(np.square(diff).sum(1))))
    return distances


# #### Question 3.1.3
# Use the `fast_distances` function provided above to compute the distance from the first movie in the test set to all the movies in the training set, **using your set of 20 features**.  Make a new table called `genre_and_distances` with one row for each movie in the training set and two columns:
# * The `"Genre"` of the training movie
# * The `"Distance"` from the first movie in the test set 
# 
# Ensure that `genre_and_distances` is **sorted in increasing order by distance to the first test movie**.

# In[40]:


# The staff solution took 4 lines of code.

train_20 = train_movies.select(my_20_features)
test_20 = test_movies.select(my_20_features)
test_20_row = test_20.row(0)

a = Table().with_column(
    'Genre', train_movies.column('Genre'), 
    'Distance', fast_distances(test_20_row, train_20)
)


genre_and_distances = a.sort('Distance')
genre_and_distances


# In[ ]:


_ = ok.grade("q3_1_3")
_ = ok.backup()


# #### Question 3.1.4
# Now compute the 5-nearest neighbors classification of the first movie in the test set.  That is, decide on its genre by finding the most common genre among its 5 nearest neighbors, according to the distances you've calculated.  Then check whether your classifier chose the right genre.  (Depending on the features you chose, your classifier might not get this movie right, and that's okay.)

# In[41]:


# Set my_assigned_genre to the most common genre among these.
my_assigned_genre = genre_and_distances.group('Genre').sort('count', True).column('Genre').item(0)

# Set my_assigned_genre_was_correct to True if my_assigned_genre
# matches the actual genre of the first movie in the test set.
my_assigned_genre_was_correct = my_assigned_genre == test_movies.column('Genre').item(0)

print("The assigned genre, {}, was{}correct.".format(my_assigned_genre, " " if my_assigned_genre_was_correct else " not "))


# In[ ]:


_ = ok.grade("q3_1_4")
_ = ok.backup()


# ## 3.2. A classifier function
# 
# Now we can write a single function that encapsulates the whole process of classification.

# #### Question 3.2.1
# Write a function called `classify`.  It should take the following four arguments:
# * A row of features for a movie to classify (e.g., `test_20.row(0)`).
# * A table with a column for each feature (e.g., `train_20`).
# * An array of classes that has as many items as the previous table has rows, and in the same order.
# * `k`, the number of neighbors to use in classification.
# 
# It should return the class a `k`-nearest neighbor classifier picks for the given row of features (the string `'Romance'` or the string `'Action'`).

# In[56]:


def classify(test_row, train_rows, train_classes, k):
    """Return the most common class among k nearest neigbors to test_row."""
    distances = fast_distances(test_row, train_rows)
    genre_and_distances = Table(). with_column('Genre', train_classes, 'Distance', distances)
    genre_and_distances_sort = genre_and_distances.sort('Distance')
    top_k = genre_and_distances_sort.take(np.arange(k))
    top_k = top_k.group('Genre').sort('count', True) 
    return top_k.column('Genre').item(0)


# In[57]:


_ = ok.grade("q3_2_1")
_ = ok.backup()


# #### Question 3.2.2
# 
# Assign `king_kong_genre` to the genre predicted by your classifier for the movie "king kong" in the test set, using **11 neighbors** and using your 20 features.

# In[58]:


# The staff solution first defined a row object called king_kong_features.
king_kong_features = test_movies.where('Title', "king kong").select(my_20_features).row(0)
train_movies_20 = train_movies.select(my_20_features)
train_classes = train_movies.column('Genre')

king_kong_genre = classify(king_kong_features, train_movies_20, train_classes, 11)
king_kong_genre


# In[47]:


_ = ok.grade("q3_2_2")
_ = ok.backup()


# Finally, when we evaluate our classifier, it will be useful to have a classification function that is specialized to use a fixed training set and a fixed value of `k`.

# #### Question 3.2.3
# Create a classification function that takes as its argument a row containing your 20 features and classifies that row using the 11-nearest neighbors algorithm with `train_20` as its training set.

# In[48]:


def classify_one_argument(row):
    return classify(row, train_movies_20, train_classes, 11)

# When you're done, this should produce 'Romance' or 'Action'.
classify_one_argument(test_20.row(0))


# In[49]:


_ = ok.grade("q3_2_3")
_ = ok.backup()


# ## 3.3. Evaluating your classifier

# Now that it's easy to use the classifier, let's see how accurate it is on the whole test set.
# 
# **Question 3.3.1.** Use `classify_one_argument` and `apply` to classify every movie in the test set.  Name these guesses `test_guesses`.  **Then**, compute the proportion of correct classifications. 

# In[50]:


test_attributes = test_movies.select(my_20_features)

test_guesses = test_attributes.apply(classify_one_argument)
combine = test_movies.with_column("guess", test_guesses).select('guess', 'Genre')


proportion_correct = (np.count_nonzero(combine.column("guess") == combine.column('Genre'))) / test_movies.num_rows
proportion_correct


# In[51]:


_ = ok.grade("q3_3_1")
_ = ok.backup()


# At this point, you've gone through one cycle of classifier design.  Let's summarize the steps:
# 1. From available data, select test and training sets.
# 2. Choose an algorithm you're going to use for classification.
# 3. Identify some features.
# 4. Define a classifier function using your features and the training set.
# 5. Evaluate its performance (the proportion of correct classifications) on the test set.

# ## 4. Extra Explorations
# Now that you know how to evaluate a classifier, it's time to build a better one.

# #### Question 4.1
# Find a classifier with better test-set accuracy than `classify_one_argument`.  (Your new function should have the same arguments as `classify_one_argument` and return a classification.  Name it `another_classifier`.)  You can use more or different features, or you can try different values of `k`.  (Of course, you still have to use `train_movies` as your training set!)

# In[52]:


# To start you off, here's a list of possibly-useful features:
# FIXME: change these example features
staff_features = make_array('afraid', 'marri', 'women', 'happi', 'command', 'captain', 'secur', 'record', 'protect', 'tonight', 'shot', 'save', 'trust', 'miss', 'husband', 'system', 'emerg', 'blood', 'war', 'secret', "heart")

train_classes = train_movies.column('Genre')
def another_classifier(row):
    return classify(row, train_staff, train_classes, 12)

train_staff = train_movies.select(staff_features)
test_staff = test_movies.select(staff_features)
test_guesses = test_staff.apply(another_classifier)
combine = test_movies.with_column("guess", test_guesses).select('guess', 'Genre')
proportion_correct = (np.count_nonzero(combine.column("guess") == combine.column('Genre'))) / test_movies.num_rows
proportion_correct


# Briefly describe what you tried to improve your classifier. As long as you put in some effort to improving your classifier and describe what you have done, you will receive full credit for this problem.

# I tried to include different words in the features to test if it would produce a higher proportion of correct. I also increased and decreased the k value.

# Congratulations: you're done with the required portion of the project! Time to submit.

# In[60]:


_ = ok.submit()


# ## Ungraded and Optional: A Custom Classifier
# Try to create an even better classifier. You're not restricted to using only word proportions as features.  For example, given the data, you could compute various notions of vocabulary size or estimated movie length.  If you're feeling very adventurous, you could also try other classification methods, like logistic regression.  If you think you have built a classifier that works well, post on Piazza and let us know or enter the completely optional Kaggle competition.

# In[ ]:


#####################
# Custom Classifier #
#####################


# In[ ]:


_ = ok.submit()


# ## Kaggle Competition
# 
# **Note:** This part is completely optional and will not contribute towards your grade in any way.
# 
# We decided to *hold out* a set of 50 movies, for which we have provided the attributes but not the genres. You can use this set to evaluate how well you classifier performs on data for which you have never seen the correct genres. Optionally, you can submit your predictions on this dataset to Kaggle to compare your classifier to other students in the class (whoever else decides to participate).
# 
# To participate, use your classifier to predict the genre of each row in the `holdout` table. Then, call
# ```create_competition_submission```
# to generate a CSV file that you can submit to the competition!
# 
# If you want to participate in the competition, you will have to create a Kaggle account. It’s easiest for the staff to determine the winners of the competition if you use your `@berkeley.edu` email when doing so, but you can also contact GSI Vasilis (v.oikonomou@berkeley.edu) if you decide to use another email address. 
# 
# The prize for the winners of this competition is dinner with the Professors! For more details, visit the competition page found below.
# 
# When you are ready to make a submission, go to https://www.kaggle.com/t/c3a4430bdb454d3293aefb4c1becd582 instructions.
# 
# **Hint: ** Remember that k-NN classifiers are susceptible to the scale of our data. For this reason, it might be a good idea to convert the columns Year, Rating, # Votes and # Words to Standard Units before using them in your classifier!

# In[ ]:


hold_out_movies = Table.read_table('proj3_test_set.csv')
hold_out_movies.show(5)


# As an example, we provide a sample of how you can generate some predictions and make a prediction file. After the file has been generated, download it and suubmit it on the competition page.

# In[ ]:


def create_competition_submission(predictions, filename='my_submission.csv'):
    '''
    Create a submission CSV for the Kaggle competition.
    Inputs: predictions - list or array of your predictions (Generated as in Question 3.3.1.)
    '''
    #Table().with_columns('Id', np.arange(len(predictions)), 'Predictions', predictions).to_csv(filename)
    print('Created', filename)
    Table().with_columns("index", np.arange(hold_out_movies.num_rows), "Genre", predictions).to_csv(filename)


# In[ ]:


# generate submissions with your classifier
random_predictions = np.random.choice(make_array("romance", "action"), hold_out_movies.num_rows)
random_predictions


# In[ ]:


# generate a submission file called "example.csv". Now you can take the file generated and submit it on Kaggle
create_competition_submission(random_predictions, "example_submission.csv")


# In[59]:


# For your convenience, you can run this cell to run all the tests at once!
import os
print("Running all tests...")
_ = [ok.grade(q[:-3]) for q in os.listdir("tests") if q.startswith('q')]
print("Finished running all tests.")

