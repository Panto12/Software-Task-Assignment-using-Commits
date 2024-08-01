import os
import json
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import hunk_info_extract

lemmatizer = WordNetLemmatizer()
htmlRegex = re.compile(r'<[^>]+>')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("english"))


# Function to preprocess a field of an object json type issue report
def clean_data(item, field):
    
    # 1. Remove numbers (replace with a blank space)
    current_text = re.sub(r'\d+', ' ', item[field])

    # 2. Split camelCased words
    current_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", current_text)

    # 3. Convert to lower case
    current_text = current_text.lower()

    # 4. Replace underscore with a blank space
    current_text = re.sub(r'_', ' ', current_text)

    # 5. Remove whitespace from text
    current_text = " ".join(current_text.split())

    # 6. Remove punctuation while tokenizing
    current_text_tokens = tokenizer.tokenize(current_text)

    # 7. Remove single-character words
    current_text_tokens = [word for word in current_text_tokens if len(word) > 1]
    
    # 8. Remove stopwords
    current_text_tokens = [word for word in current_text_tokens if word not in stop_words]

    # 9. Lemmatize string
    current_text_filter = ' '.join([lemmatizer.lemmatize(word, pos ='v') for word in current_text_tokens])
    

    return current_text_filter


# Function to preprocess the text fields of all issues in a project
def preprocess_dataset(project):
           
    # Run for every issue json file in the project           
    for issue in os.listdir("data/" + project):

        # Create string with the file path to the current issue file
        filepath = "data/" + project + "/" + issue

        # Open the issue file, read its contents, and store the JSON data in the issue_json_object            
        with open(filepath, 'r+') as issuefile:
            issue_json_object = json.load(issuefile)
                
            # Clean title of issue
            cleanedTitle = clean_data(issue_json_object, "title")
                
            # Cleaned text for issue and description to be added
            cleanedTitleDesc = cleanedTitle
                
            # Clean description of issue if exists
            if "desc" in issue_json_object:
                cleanedDescription = clean_data(issue_json_object, "desc")
                    
                # Cleaned text for issue and description to be added if description exists
                cleanedTitleDesc = cleanedTitle + " " + cleanedDescription
                
            # List for cleaned messages of every commit linked to the issue
            cleanedCommitMessage = []

            # List for code snippets of every commit linked to the issue
            cleanedCodeSnippets = []
                
            # List for the edit representation tag sequence vector of every commit linked to the issue
            edtiInfoVectors = []

            # Run for every commit linked to the issue
            for commit in issue_json_object["linked_commits"]:
                    
                # Clean message of commit
                message = clean_data(commit, "message")
                    
                # Add cleaned message to list
                cleanedCommitMessage.append(message)
                    
                # Clean content of every code snippet of commit and join all the cleaned code snippets of commit
                codeSnippets = " ".join([clean_data(codeSnippet, "content") for codeSnippet in commit["code_snippets"]])

                # Add the cleaned code snippets of commit to list    
                cleanedCodeSnippets.append(codeSnippets)

                # Get the vector of percentages from the edit of the commit
                commitEditPercVector = hunk_info_extract.sequencePercentageVector(commit)

                # Add the edit representation to list
                edtiInfoVectors.append(commitEditPercVector)


            # Cleaned data to be added in the issue json                
            new_data_dict = {"clean_title_desc": cleanedTitleDesc,
                                "clean_commit_message": cleanedCommitMessage,
                                "clean_commit_code": cleanedCodeSnippets,
                                "edits_info": edtiInfoVectors}

            # Update the issue json with the new data                                
            issue_json_object.update(new_data_dict)
            issuefile.seek(0)
            json.dump(issue_json_object, issuefile, default=str, indent = 3)