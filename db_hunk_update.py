# This file shall be run once to update the "hunk" collection in database. 
# It adds the "commit_id" field in every document so that we avoid using the 
# intermediary collection "file_action"


from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


# Connect to mongodb
client = MongoClient("localhost", 27017)

# Check the connection to the mongodb
try:
    info = client.server_info() # Forces a call.
    print("\nSuccesful connection\n")
except ServerSelectionTimeoutError:
    print("\nServer is down.\n")

# Connect to smartshark db
db = client['smartshark_2_2']
files_actions = db['file_action']
codeSnippets = db['hunk']

# Get all the file_actions
allFilesActions = files_actions.find()

# Create a dictionary to store the mapping from file_action_id to commit_id
mapping = {}

# Fill the mapping dictionary
for file_action in allFilesActions:
    mapping[file_action['_id']] = file_action['commit_id']

# Update each hunk with the commit_id
for snippet in codeSnippets.find():
    codeSnippets.update_one({'_id': snippet['_id']}, 
                                {'$set': {'commit_id': mapping.get(snippet['file_action_id'])}})
								   