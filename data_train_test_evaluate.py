import re
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from gensim.models import Word2Vec
import fasttext
from results import getProjectInfo

from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dropout

from keras import backend as K
from keras.metrics import Precision, Recall

precision = Precision()
recall = Recall()

def f1_score(y_true, y_pred):
    """Computes the F1 Score

    Args:
      y_true: true labels.
      y_pred: predicted labels.

    Returns:
      F1 score.
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))


randomStateSplit = 42


# To ignore warnings raised when the precision and F-score metrics are calculated using labels that have no predicted samples
import warnings
warnings.filterwarnings('ignore')  

# fields = ['issueId', 'dateCreated', 'dateUpdated', 'dateOfCommit', 'type', 'priority', 'components', 'labels', 
# 					'cleanTitleDesc', 'cleanCommitMessage', 'cleanCommitCode', 'reporter', 'assignee', 'committer', 'editInfoVector'] 


def topDevsChoose(df, k):

    # Count the occurrences of each ID in the 'target' column
    devCounts = df['target'].value_counts()

    # Get the top-x IDs based on their counts
    top_k_ids = devCounts.index[:k]

    # Filter the DataFrame to keep only the rows with the top-x IDs
    filtered_df = df[df['target'].isin(top_k_ids)]

    return filtered_df


def targetClassChoose(projectName):

    # Read in csv file and sort by column of date
    df = pd.read_csv("data_csv/" + projectName + ".csv").sort_values(by='dateCreated')
    
    # Convert columns values from string to datetime objects
    df['dateCreated'] = pd.to_datetime(df['dateCreated'])
    df['dateUpdated'] = pd.to_datetime(df['dateUpdated'])

    # Create a new column in the df to store the chosen as target developers and the corresponding commit-to-update date difference
    df['target'] = None
    df['dateDiff'] = None
    
    # Variables for the developer stats about the target choice
    case1 = case2 = case3_1 = case3_2 = case4 = 0

    # Developer choosing for each issue as target (for each row of the df)
    for index, row in df.iterrows():
        
        # Convert the string representation of the list of committers and the commit dates to a list
        committersList = ast.literal_eval(row['committer'])
        commitDateList = ast.literal_eval(row['dateOfCommit'])

        # Convert the string representations of the lists to lists
        row['cleanCommitMessage'] = ast.literal_eval(row['cleanCommitMessage']) 
        row['cleanCommitCode'] = ast.literal_eval(row['cleanCommitCode']) 
        row['editInfoVector'] = ast.literal_eval(row['editInfoVector']) 

        for i in range(len(row['editInfoVector'])):
            if row['editInfoVector'][i] == 0:
                row['editInfoVector'][i] = [0, 0, 0, 0, 0]

        # List for days difference from commit date to update date 
        commitToUpdate = []

        # Find the dates differences and add them to the commitToUpdate list
        for date in commitDateList:
            dateDifference = (row['dateUpdated'] - pd.to_datetime(date)).days
            commitToUpdate.append(dateDifference)


        # Check the conditions and update the target column

        # Case 1: committer is also assignee
        if row['assignee'] in committersList:

            # Get the index of the assignee in the list of committers
            index_case_1 = committersList.index(row['assignee'])

            # Assign as target the assignee-committer 
            df.at[index, 'target'] = row['assignee']

            # Update corresponding values to cleanCommitMessage, cleanCommitCode, and editRepresentation columns
            df.at[index, 'cleanCommitMessage'] = row['cleanCommitMessage'][index_case_1]
            df.at[index, 'cleanCommitCode'] = row['cleanCommitCode'][index_case_1]
            df.at[index, 'editInfoVector'] = row['editInfoVector'][index_case_1]
            df.at[index, 'dateDiff'] = commitToUpdate[index_case_1]

            # Add 1 for every developer found for case 1
            case1 = case1 + 1

        # Case 2: committer is not assignee but is reporter            
        elif row['reporter'] in committersList:
            
            # Get the index of the reporter in the list of committers
            index_case_2 = committersList.index(row['reporter'])

            # Assign as target the reporter-committer 
            df.at[index, 'target'] = row['reporter']

            # Update corresponding values to cleanCommitMessage, cleanCommitCode, and editRepresentation columns
            df.at[index, 'cleanCommitMessage'] = row['cleanCommitMessage'][index_case_2]
            df.at[index, 'cleanCommitCode'] = row['cleanCommitCode'][index_case_2]
            df.at[index, 'editInfoVector'] = row['editInfoVector'][index_case_2]
            df.at[index, 'dateDiff'] = commitToUpdate[index_case_2]
            
            # Add 1 for every developer found for case 2
            case2 = case2 + 1


        # Case 3: Assignee exists but is not a committer and reporter is not a committer
        elif row['assignee'] != '0':

            # Case 3.1: Days from commit to update is 1 or less            
            if any(0 <= d <= 1 for d in commitToUpdate):
                
                # Find the first index of the element that satisfies the commitToUpdate restriction 
                index_case_3_1 = next(i for i, d in enumerate(commitToUpdate) if 0 <= d <= 1)
                
                # Assign as target the committer indicated by the index found
                df.at[index, 'target'] = committersList[index_case_3_1]

                # Update corresponding values to cleanCommitMessage, cleanCommitCode, and editRepresentation columns
                df.at[index, 'cleanCommitMessage'] = row['cleanCommitMessage'][index_case_3_1]
                df.at[index, 'cleanCommitCode'] = row['cleanCommitCode'][index_case_3_1]
                df.at[index, 'editInfoVector'] = row['editInfoVector'][index_case_3_1]
                df.at[index, 'dateDiff'] = commitToUpdate[index_case_3_1]

                # Add 1 for every developer found for case 3_1
                case3_1 = case3_1 + 1


            # Case 3.2: Days from commit to update is more than 1
            else:

                # Find the index of the element with the minimum positive value in commitToUpdate, or maximum negative value if all values are negative
                min_pos = None
                max_neg = None
                for i, val in enumerate(commitToUpdate):
                    if val >= 0 and (min_pos is None or val < commitToUpdate[min_pos]):
                        min_pos = i
                    elif val < 0 and (max_neg is None or val > commitToUpdate[max_neg]):
                        max_neg = i

                if min_pos is not None:
                    index_case_3_2 = min_pos
                else:
                    index_case_3_2 = max_neg


                # Assign as target the assignee
                df.at[index, 'target'] = row['assignee']

                # Update corresponding values to cleanCommitMessage, cleanCommitCode, and editRepresentation columns
                df.at[index, 'cleanCommitMessage'] = row['cleanCommitMessage'][index_case_3_2]
                df.at[index, 'cleanCommitCode'] = row['cleanCommitCode'][index_case_3_2]
                df.at[index, 'editInfoVector'] = row['editInfoVector'][index_case_3_2]
                df.at[index, 'dateDiff'] = commitToUpdate[index_case_3_2]
    
                # Add 1 for every developer found for case 3_2
                case3_2 = case3_2 + 1


        # Case 4: Assignee does not exist and reporter is not a committer
        else:
            
            # Find the index of the element with the minimum positive value in commitToUpdate, or maximum negative value if all values are negative
            min_pos = None
            max_neg = None
            for i, val in enumerate(commitToUpdate):
                if val >= 0 and (min_pos is None or val < commitToUpdate[min_pos]):
                    min_pos = i
                elif val < 0 and (max_neg is None or val > commitToUpdate[max_neg]):
                    max_neg = i

            if min_pos is not None:
                index_case_4 = min_pos
            else:
                index_case_4 = max_neg

            # Assign as target the corresponding committer from committersList using the index
            df.at[index, 'target'] = committersList[index_case_4]

            # Update corresponding values to cleanCommitMessage, cleanCommitCode, and editRepresentation columns
            df.at[index, 'cleanCommitMessage'] = row['cleanCommitMessage'][index_case_4]
            df.at[index, 'cleanCommitCode'] = row['cleanCommitCode'][index_case_4]
            df.at[index, 'editInfoVector'] = row['editInfoVector'][index_case_4]
            df.at[index, 'dateDiff'] = commitToUpdate[index_case_4]

            # Add 1 for every developer found for case 4
            case4 = case4 + 1


    # Define a dictionary to keep track of the number of issues in each case
    issuesPerCase = {
    'Case 1': case1,
    'Case 2': case2,
    'Case 3.1': case3_1,
    'Case 3.2': case3_2,
    'Case 4': case4
    }

    # Threshold for least appearances of the target developer
    devThreshold = 15

    # Drop rows with empty string or only blank spaces in cleanCommitCode column
    df = df[df['cleanCommitCode'].str.strip().str.len() > 0]

    # Drop rows where there is no priority level assigned
    df = df[df['priority'] != '0']

    # Number of appearances for each target developer of the project
    targetDevCounts = df['target'].value_counts()

    # Find the indexes of the target developers that have less appearances than the threshold
    rowsToDrop = targetDevCounts[targetDevCounts < devThreshold].index

    # Drop rows from the df that are indicated by the indexes found
    df.drop(df[df['target'].isin(rowsToDrop)].index, inplace=True)


    return df, issuesPerCase


# Function to split data to training and test sets and evaluate the model
def train_and_evaluate(projectName):

    # Call the function targetClassChoose() to get the dataframe and issuesPerCase variable
    df, issuesPerCase = targetClassChoose(projectName)

    # Filter the df keeping the top-k devs
    df = topDevsChoose(df, 2)

    # Get the number of issues in the dataframe
    issuesNumber = len(df)

    # Count the occurrences of each ID in the 'target' column
    devCounts = df['target'].value_counts().values

    # Extract the columns of data and target class
    X_title_desc = df['cleanTitleDesc']
    X_type = df['type']
    X_priority = df['priority']
    X_components = df['components']
    X_labels = df['labels']
    X_commit_message = df['cleanCommitMessage']
    X_commit_code = df['cleanCommitCode']
    X_edits_info = df['editInfoVector']
    X_date_diff = df['dateDiff']

    y = df['target']

    # Feature TitleDesc with TF-IDF --------------------------------------- 

    # Textual feature of title and description of issue report - TF-IDF
    tfidf_vectorizer_1 = TfidfVectorizer()
    featureTitleDesc = tfidf_vectorizer_1.fit_transform(X_title_desc)

    # Convert feature to array
    featureTitleDesc = featureTitleDesc.toarray()

    # Feature Type -------------------------------------------------------
    
    # Reshape the issue type data from 1D to 2D
    X_type = X_type.values.reshape(-1, 1)

    # Categorical feature of type of issue - OneHotEncoder
    type_encoder = OneHotEncoder()
    featureType = type_encoder.fit_transform(X_type)
    
    # Convert feature to array
    featureType = featureType.toarray()

    # Feature Priority ----------------------------------------------------

    # Define the ordered categories of priority
    priority_categories = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']

    # Initialize the ordinal encoder
    ordinal_encoder = OrdinalEncoder(categories=[priority_categories])

    # Reshape the priority variable from 1D to 2D
    X_priority = X_priority.values.reshape(-1, 1)

    # Perform ordinal encoding
    featurePriority = ordinal_encoder.fit_transform(X_priority)

    # Feature Components --------------------------------------------------

    # Check if components tags exist
    if not (all(x == 0 for x in X_components) or all(x == '0' for x in X_components)):
        
        # Preprocess the components to be encoded (get rid of needless quotation marks) and lowercase the words
        X_components = [[word.lower() for word in re.findall("'(\w+)'", component)] for component in X_components]

        # Multi-label categorical feature of components for issue report - MultiLabelBinarizer
        components_encoder = MultiLabelBinarizer()
        featureComponents = components_encoder.fit_transform(X_components)

        # Set variable to True if components exist
        featureComponentsUse = True

    else:
        
        # Set variable to False if components don't exist
        featureComponentsUse = False
        
    # Feature Labels ------------------------------------------------------

    # Check if labels tags exist
    if not (all(x == 0 for x in X_labels) or all(x == '0' for x in X_labels)):

        # Preprocess the labels to be encoded (get rid of needless quotation marks) and lowercase the words
        X_labels = [[word.lower() for word in re.findall("'(\w+)'", label)] for label in X_labels]

        # Multi-label categorical feature of labels for issue report - MultiLabelBinarizer
        labels_encoder = MultiLabelBinarizer()
        featureLabels = labels_encoder.fit_transform(X_labels)
    
        # Set variable to True if labels exist
        featureLabelsUse = True

    else:

        # Set variable to False if labels don't exist
        featureLabelsUse = False

    # Feature Commit Message with TF-IDF ----------------------------------

    # Textual feature of the message of the commit(s) linked to the issue - TF-IDF
    tfidf_vectorizer_2 = TfidfVectorizer()
    featureCommitMessage = tfidf_vectorizer_2.fit_transform(X_commit_message)

    # Convert feature to array
    featureCommitMessage = featureCommitMessage.toarray()


    # Feature Commit Code with TF-IDF -------------------------------------

    # Textual feature of the code of the commit(s) linked to the issue - TF-IDF
    tfidf_vectorizer_3 = TfidfVectorizer()
    featureCommitCode = tfidf_vectorizer_3.fit_transform(X_commit_code)

    # Convert feature to array
    featureCommitCode = featureCommitCode.toarray()

    # Feature Commit Code for embedding -----------------------------------

    featureCommitCodeEmbed = np.array(X_commit_code)

    # Feature Edit Info ---------------------------------------------------

    # Convert the feature X_edits_info to a pandas DataFrame, and transpose it
    feature_transposed = pd.DataFrame(X_edits_info.tolist()).transpose()

    # Initialize an empty list for the scaled feature rows
    scaled_rows = []

    # Iterate over each row of the transposed feature DataFrame
    for i in range(len(feature_transposed)):

        # Get the ith row of the feature DataFrame
        row = feature_transposed.iloc[i]
        
        # Convert the row to a 2D numpy array
        row_np = np.array(row)
        row_np = row_np.reshape(-1, 1)

        # Create a StandardScaler object for the row
        row_scaler = StandardScaler()

        # Scale the row using the StandardScaler object
        scaled_row_np = row_scaler.fit_transform(row_np)
        
        # Flatten the scaled row to a 1D numpy array
        scaled_row = scaled_row_np.flatten()

        # Append the scaled row to the list of scaled rows
        scaled_rows.append(scaled_row)

    # Convert the list of scaled features to a 2D numpy array and transpose it
    scaled_rows = np.array(scaled_rows).T

    # Create a new DataFrame with the scaled feature and the original index
    feature_scaled = pd.DataFrame(scaled_rows, index=X_edits_info.index)

    # Convert the DataFrame to a numpy array
    featureEditInfo = np.array(feature_scaled)

    # Feature Date Diff ---------------------------------------------------

    # Define bin edges
    bin_edges = [-np.inf, -2, 1, 10, 30, np.inf]

    # Use digitize() to assign each element to a bin
    featureDateDiff = np.digitize(X_date_diff, bin_edges)

    # Reshape the featurePriority array from 1D to 2D
    featureDateDiff = featureDateDiff.reshape(-1, 1)
     
    # Target Class --------------------------------------------------------

    # Reshape the target variable for one-hot encoding
    y = y.values.reshape(-1, 1)

    # Encode the target variable using one-hot encoding
    target_encoder = OneHotEncoder()
    target_class_y = target_encoder.fit_transform(y)

    # Convert feature to array
    target_class_y = target_class_y.toarray()

    # Get the number of different components
    num_classes = target_class_y.shape[1]

    # ------------------------------------------------------------------------------------------

    # Get Word2Vec embeddings for issue reports in training and testing sets
    def word2VecEmbeddings(model, set, size):

        # Create an empty list to store the word2vec embeddings for each text in the input set
        w2vList = []

        # Iterate over each text in the input set
        for text in set:

            # For each word in the text, get its corresponding vector representation from the trained word2vec model
            # Only include words that have a vector representation in the model's vocabulary
            vectors = [model.wv[word] for word in text.split() if word in model.wv]

            # If any valid word vectors were found, take their element-wise mean to get a single vector
            if vectors:
                w2vList.append(np.mean(vectors, axis=0))
            
            # If no valid word vectors were found (e.g., if the text consists entirely of out-of-vocabulary words),
            # add a zero vector to the list as a default
            else:
                w2vList.append(np.zeros(size))

        # Convert the list of embeddings to a numpy array and return it
        w2vList = np.array(w2vList)

        return w2vList 
    

    # Function to get the necessary pieces for the neural netowrk from a feature
    def nnModelFeature(feature, y, featureName, method=None):

        # Split the feature and y data into training and testing sets
        trainingSet, testingSet, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=randomStateSplit)

        # If we want to apply Word2Vec on the feature
        if method == "word2vec":

            # Tokenize the feature by splitting it into sentences and words
            sentences = [text.split() for text in feature]
            words = [word for sentence in sentences for word in sentence]

            # Get the set of unique words and count the number of it
            uniqueWords = set(words)
            num_unique_words = len(uniqueWords)

            # Set the vector size of Word2Vec based on the number of unique words in the trainingSet
            if num_unique_words < 2000:
                sizeWord2Vec = 200
            elif num_unique_words < 5000:
                sizeWord2Vec = 250
            else:
                sizeWord2Vec = 300
    
            # Train the Word2Vec model on the tokenized feature
            model_word2vec = Word2Vec(sentences=sentences, vector_size=sizeWord2Vec, window=3, min_count=10, workers=4)

            # Embed the training and testing sets using the trained Word2Vec model
            trainingSet = word2VecEmbeddings(model_word2vec, trainingSet, sizeWord2Vec)
            testingSet = word2VecEmbeddings(model_word2vec, testingSet, sizeWord2Vec)

            # Define the input layer for the neural network with the size of the Word2Vec embeddings
            input = keras.Input(shape=(sizeWord2Vec,), name=featureName)
        
        # If we want to apply FastText on the feature
        elif method == "fastText":
            
            # Assign the appropriate trainingFilename variable based on the input feature name
            if featureName == 'title_desc_input_ft':
                trainingFilename = 'trainingTitleDesc.txt'

            elif featureName == 'commit_message_input_ft':
                trainingFilename = 'trainingCommitMessage.txt'

            elif featureName == 'commit_code_input_ft':
                trainingFilename = 'trainingCommitCode.txt'

            # Save the commit code feature from the training set to a text file
            with open(trainingFilename, 'w', encoding='utf-8') as ft_file:
                for feature_text in trainingSet:
                    ft_file.write(feature_text + '\n')

            # Find the number of unique words in the training set
            unique_words = set()
            for commit_code in trainingSet:
                words = commit_code.split()
                unique_words.update(words)
            num_unique_words = len(unique_words)

            # Set the vector size of FastText based on the number of unique words in the trainingSet
            if num_unique_words < 2000:
                sizeFastText = 100
            elif num_unique_words < 5000:
                sizeFastText = 150
            else:
                sizeFastText = 200

            # Train FastText on the training set
            model_ft = fasttext.train_unsupervised(trainingFilename, model='skipgram', wordNgrams=1, minn=2, maxn=8, dim=sizeFastText)
            
            # Get the FastText embeddings for the training and testing sets
            trainingSet = np.array([model_ft.get_sentence_vector(text) for text in trainingSet])
            testingSet = np.array([model_ft.get_sentence_vector(text) for text in testingSet])

            # Define the input layer for the neural network with the size of the FastText embeddings
            input = keras.Input(shape=(sizeFastText,), name=featureName)

        else:

            # Define the input layer for the neural network with the size of the number of columns in the trainingSet
            input = keras.Input(shape=(trainingSet.shape[1],), name=featureName)

        # Define a dense layer with 64 neurons and a ReLU activation function. The layer takes the input layer as its input.
        x = keras.layers.Dense(64, activation='relu')(input)

        # Return x (the output from the dense layer), input (the input layer) and the training and testing sets
        return x, input, trainingSet, testingSet, y_train, y_test

       

    # Call the nnModelFeature function for each of the features

    # 1. Title and Description feature
    x1, input_title_desc, X_title_desc_train, X_title_desc_test, y_train, y_test = nnModelFeature(featureTitleDesc, target_class_y, 'title_desc_input')
    
    # 2. Type feature
    x2, input_type, X_type_train, X_type_test, _, _ = nnModelFeature(featureType, target_class_y, 'type_input')
    
    # 3. Priority feature
    x3, input_priority, X_priority_train, X_priority_test, _, _ = nnModelFeature(featurePriority, target_class_y, 'priority_input')

    # 4. Components feature - if exists, else set to None
    if featureComponentsUse:
        x4, input_components, X_components_train, X_components_test, _, _ = nnModelFeature(featureComponents, target_class_y, 'components_input')
    else:
        x4 = input_components = X_components_train = X_components_test = None
    
    # 5. Labels feature - if exists, else set to None    
    if featureLabelsUse:
        x5, input_labels, X_labels_train, X_labels_test, _, _ = nnModelFeature(featureLabels, target_class_y, 'labels_input')
    else:
        x5 = input_labels = X_labels_train = X_labels_test = None
    
    # 6. Commit Message feature    
    x6, input_commit_message, X_commit_message_train, X_commit_message_test, _, _ = nnModelFeature(featureCommitMessage, target_class_y, 'commit_message_input')
    
    # 7. Commit Code feature
    x7, input_commit_code, X_commit_code_train, X_commit_code_test, _, _ = nnModelFeature(featureCommitCode, target_class_y, 'commit_code_input')

    # 8. Edit Info feature
    x8, input_edit_info, X_edit_info_train, X_edit_info_test, _, _ = nnModelFeature(featureEditInfo, target_class_y, 'edit_info_input')

    # 9. Date Diff feature
    x9, input_date_diff, X_date_diff_train, X_date_diff_test, _, _ = nnModelFeature(featureDateDiff, target_class_y, 'date_diff_input')

    # 1.2. Wor2Vec Title and Description feature
    x1_w2v, input_title_desc_w2v, X_title_desc_train_w2v, X_title_desc_test_w2v, _, _ = nnModelFeature(X_title_desc, target_class_y, 'title_desc_input_w2v', "word2vec")

    # 6.2. Wor2Vec Commit Message feature
    x6_w2v, input_commit_message_w2v, X_commit_message_train_w2v, X_commit_message_test_w2v, _, _ = nnModelFeature(X_commit_message, target_class_y, 'commit_message_input_w2v', "word2vec")

    # 7.2. Wor2Vec Commit Code feature
    x7_w2v, input_commit_code_w2v, X_commit_code_train_w2v, X_commit_code_test_w2v, _, _ = nnModelFeature(featureCommitCodeEmbed, target_class_y, 'commit_code_input_w2v', "word2vec")
    
    # 1.3. FastText Title and Description feature
    x1_ft, input_title_desc_ft, X_title_desc_train_ft, X_title_desc_test_ft, _, _ = nnModelFeature(X_title_desc, target_class_y, 'title_desc_input_ft', "fastText")

    # 6.3. FastText Commit Message feature
    x6_ft, input_commit_message_ft, X_commit_message_train_ft, X_commit_message_test_ft, _, _ = nnModelFeature(X_commit_message, target_class_y, 'commit_message_input_ft', "fastText")

    # 7.3. FastText Commit Code feature
    x7_ft, input_commit_code_ft, X_commit_code_train_ft, X_commit_code_test_ft, _, _ = nnModelFeature(featureCommitCodeEmbed, target_class_y, 'commit_code_input_ft', "fastText")
    


    # Function that creates, compiles, trains, and evaluates a neural network model
    def nnModel(mergedFeatures, inputFeatures, trainFeatures, testFeatures, hidden_size=128):

        # Check if there is more than one feature to be merged
        if len(mergedFeatures) > 1:

            # Concatenate (merge) the features from mergedFeatures into a single layer
            merged_model = keras.layers.concatenate(mergedFeatures)

        # For the case that one feature has been passed to the function
        else:
            
            # Get the only feature from mergedFeatures
            merged_model = mergedFeatures[0]

        # Add a hidden layer with the specified size
        hidden_layer = keras.layers.Dense(hidden_size, activation='relu')(merged_model)
        
        # Add a dropout layer
        dropout_layer = Dropout(0.3)(hidden_layer)

        # Define output of neural network with softmax
        output_model = keras.layers.Dense(num_classes, activation='softmax')(dropout_layer)

        # Define model
        model = keras.Model(inputs=inputFeatures, outputs=output_model)

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision, recall, f1_score])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=6) 

        # Train model with early stopping callback
        history_model = model.fit(trainFeatures, y_train, epochs=20, batch_size=32, validation_data=(testFeatures, y_test), callbacks=[early_stopping])

        # Evaluate model
        metricsResults = model.evaluate(testFeatures, y_test, verbose=0)

        # Return the loss and accuracy values of the model
        return metricsResults


    # Define list of models, each containing information on input, merged and split training and testing features
    models = [
        {
            # Model A1 - TitleDesc feature
            'mergedFeatures': [x1],
            'inputFeatures': [input_title_desc],
            'trainFeatures': [X_title_desc_train],
            'testFeatures': [X_title_desc_test]
        },
        {
            # Model A2 - Type, Priority, Components, Labels features
            'mergedFeatures': [x for x in [x2, x3, x4, x5] if x is not None],
            'inputFeatures': [inputF for inputF in [input_type, input_priority, input_components, input_labels] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_type_train, X_priority_train, X_components_train, X_labels_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_type_test, X_priority_test, X_components_test, X_labels_test] if testF is not None]
        },
        {
            # Model A3 - Commit Message feature
            'mergedFeatures': [x6],
            'inputFeatures': [input_commit_message],
            'trainFeatures': [X_commit_message_train],
            'testFeatures': [X_commit_message_test]
        },
        {
            # Model A4 - Commit Code feature
            'mergedFeatures': [x7],
            'inputFeatures': [input_commit_code],
            'trainFeatures': [X_commit_code_train],
            'testFeatures': [X_commit_code_test]
        },
        {
            # Model A5 - Edit Info feature
            'mergedFeatures': [x8],
            'inputFeatures': [input_edit_info],
            'trainFeatures': [X_edit_info_train],
            'testFeatures': [X_edit_info_test]
        },
        {
            # Model A6 - Date Diff feature
            'mergedFeatures': [x9],
            'inputFeatures': [input_date_diff],
            'trainFeatures': [X_date_diff_train],
            'testFeatures': [X_date_diff_test]
        },
        {
            # Model B1 - TitleDesc, Type, Priority, Components, Labels features
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_priority, input_type, input_components, input_labels] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_priority_train, X_type_train, X_components_train, X_labels_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_priority_test, X_type_test, X_components_test, X_labels_test] if testF is not None]
        },
        {
            # Model B2 - TitleDesc, Type, Priority, Components, Labels, Commit Message features
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5, x6] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_type, input_priority, input_components, input_labels, input_commit_message] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test] if testF is not None]
        },
        {
            # Model B3 - TitleDesc, Type, Priority, Components, Labels, Commit Message, Commit Code features
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5, x6, x7] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_type, input_priority, input_components, input_labels, input_commit_message, input_commit_code] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train, X_commit_code_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test, X_commit_code_test] if testF is not None]
        },
        {
            # Model B4 - TitleDesc, Type, Priority, Components, Labels, Commit Message, Commit Code, Edit Info features
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5, x6, x7, x8] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_type, input_priority, input_components, input_labels, input_commit_message, input_commit_code, input_edit_info] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train, X_commit_code_train, X_edit_info_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test, X_commit_code_test, X_edit_info_test] if testF is not None]
        },
        {
            # Model B5 - TitleDesc, Type, Priority, Components, Labels, Commit Message, Commit Code, DateDiff features
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5, x6, x7, x9] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_type, input_priority, input_components, input_labels, input_commit_message, input_commit_code, input_date_diff] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train, X_commit_code_train, X_date_diff_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test, X_commit_code_test, X_date_diff_test] if testF is not None]
        },
        {
            # Model C1 - All features with TF-IDF
            'mergedFeatures': [x for x in [x1, x2, x3, x4, x5, x6, x7, x8, x9] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc, input_type, input_priority, input_components, input_labels, input_commit_message, input_commit_code, input_edit_info, input_date_diff] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train, X_commit_code_train, X_edit_info_train, X_date_diff_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test, X_commit_code_test, X_edit_info_test, X_date_diff_test] if testF is not None]
        },
        {
            # Model C2 - All features with Word2Vec
            'mergedFeatures': [x for x in [x1_w2v, x2, x3, x4, x5, x6_w2v, x7_w2v, x8, x9] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc_w2v, input_type, input_priority, input_components, input_labels, input_commit_message_w2v, input_commit_code_w2v, input_edit_info, input_date_diff] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train_w2v, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train_w2v, X_commit_code_train_w2v, X_edit_info_train, X_date_diff_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test_w2v, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test_w2v, X_commit_code_test_w2v, X_edit_info_test, X_date_diff_test] if testF is not None]
        },
        {
            # Model C3 - All features with FastText
            'mergedFeatures': [x for x in [x1_ft, x2, x3, x4, x5, x6_ft, x7_ft, x8, x9] if x is not None],
            'inputFeatures': [inputF for inputF in [input_title_desc_ft, input_type, input_priority, input_components, input_labels, input_commit_message_ft, input_commit_code_ft, input_edit_info, input_date_diff] if inputF is not None],
            'trainFeatures': [trainF for trainF in [X_title_desc_train_ft, X_type_train, X_priority_train, X_components_train, X_labels_train, X_commit_message_train_ft, X_commit_code_train_ft, X_edit_info_train, X_date_diff_train] if trainF is not None],
            'testFeatures': [testF for testF in [X_title_desc_test_ft, X_type_test, X_priority_test, X_components_test, X_labels_test, X_commit_message_test_ft, X_commit_code_test_ft, X_edit_info_test, X_date_diff_test] if testF is not None]
        }
    ]

    # Create an empty list to store the performance of each model
    modelsPerformance = []

    # For each model in the list, train and evaluate the neural network. Append the results in the modelsPerformance list
    for model in models:
        results = nnModel(**model)
        modelsPerformance.append(results)

    # Use the getProjectInfo function to return the basic project info
    projectInfo = getProjectInfo(projectName, issuesNumber, num_classes, issuesPerCase, featureComponentsUse, featureLabelsUse)


    return projectInfo, devCounts, modelsPerformance