import os
import json
import re
import csv
import shutil
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


# Connect to mongodb
client = MongoClient("localhost", 27017)

# Check the connection to the mongodb
try:
    info = client.server_info() # Forces a call.
    print("\nSuccesful connection to database\n")
except ServerSelectionTimeoutError:
    print("\nServer is down.\n")

# Connect to smartshark db
db = client['smartshark_2_2']
projects = db['project']
issue_systems = db['issue_system']
issues = db['issue']
commits = db['commit']
codeSnippets = db['hunk']

# Read all projects and issue systems in a dictionary
allProjects = dict([(str(project["_id"]), project["name"]) for project in projects.find({})])
allIssueSystems = dict([(str(issuesystem["_id"]), str(issuesystem["project_id"])) for issuesystem in issue_systems.find({})])

# Function to get every issue we want from the database in a json file
def dataToJsonFiles():

	for issue in issues.find({}):
			
		# We want issues only if they have a title and they have been marked as Fixed
		if ( ("title" in issue) and ("resolution" in issue) and (issue["resolution"] == 'Fixed') ):
		
			# Find and add project name of issue
			projectIdOfIssue = allIssueSystems[str(issue["issue_system_id"])]
			projectNameOfIssue = allProjects[projectIdOfIssue]
			issue["project_name"] = projectNameOfIssue

			# Create array for the commits linked to the issue
			issue["linked_commits"] = []

			# Find commits of issue (the ones that fix it) and add it the issue linked commits
			for commit in commits.find({'fixed_issue_ids': issue["_id"]}):
				
				issue["linked_commits"].append(commit)

				# Create array for the code snippets of the commit
				issue["linked_commits"][-1]["code_snippets"] = []

				# Find and add the code snippets of the commit related to the issue (only the ones that have real content)
				for codeSnippet in codeSnippets.find({'commit_id': commit["_id"]}):
					# Check and include code snippet only if it contains something other than "+", "-", "*", "\n" and blank spaces
					if re.search("[^\-\+\*\n ]", codeSnippet["content"]):
						issue["linked_commits"][-1]["code_snippets"].append(codeSnippet)

			# Write issue to file only if it has at least one commit
			if (len(issue["linked_commits"]) > 0): 

				# Create directory for project
				if not os.path.exists("data/" + projectNameOfIssue):
					os.makedirs("data/" + projectNameOfIssue)

				# Add issue json file to the project directory
				with open("data/" + projectNameOfIssue + "/" + str(issue["_id"]) + ".json", 'w') as outfile:
					json.dump(issue, outfile, default=str, indent = 3)

# Function to convert the data from the json files to csv for all selected projects
def dataToCsv(projectsList):

	# Run for every project in the projects list
	for projectName in projectsList[0]:

		# Create csv file for the project
		filename = r"data_csv/" + projectName + ".csv" # 'r' prefix is used to avoid permission error

		# Create array for rows of data
		rows = []

		# Components to keep from the issue data json file
		fields = ['issueId', 'dateCreated', 'dateUpdated', 'dateOfCommit', 'type', 'priority', 'components', 'labels', 
					'cleanTitleDesc', 'cleanCommitMessage', 'cleanCommitCode', 'reporter', 'assignee', 'committer', 'editInfoVector'] 
		cols = len(fields)

		# Run for every issue in the project directory
		for issue in os.listdir("data/" + projectName):

			# Create list for the issue data    
			newRow = [0]*cols

			# Keep values of the fields we want (fields array)
			with open("data/" + projectName + "/" + issue) as json_file:
				json_data = json.load(json_file)

				# Issue Id
				newRow[0] = str(json_data["_id"])
					
				# Date the issue report was created
				newRow[1] = json_data["created_at"]
					
				# Date the issue report was updated (or closed)
				newRow[2] = json_data["updated_at"]

				# Dates when all of the commits linked to the issue report were created
				newRow[3] = [commit["committer_date"] for commit in json_data["linked_commits"]]

				# Type of issue (if exists)
				if "issue_type" in json_data:
					newRow[4] = json_data["issue_type"]

				# Priortiy of issue (if exists)
				if "priority" in json_data:
					newRow[5] = json_data["priority"]
					
				# Component tags of issue (if exist)
				if ("components" in json_data) and (json_data["components"] != []):
					newRow[6] = json_data["components"]
					
				# Label tags of issue (if exist)
				if ("labels" in json_data) and (json_data["labels"] != []):
					newRow[7] = json_data["labels"]

				# Cleaned text for title and description of issue report					
				newRow[8] = json_data["clean_title_desc"]
					
				# Cleaned text for the messages of the commits linked to issue report
				newRow[9] = json_data["clean_commit_message"]
					
				# Cleaned text for the code snippets of the commits linked to issue report
				newRow[10] = json_data["clean_commit_code"]

				# Reporter Id
				newRow[11] = str(json_data["reporter_id"])
					
				# Assignee Id (if exists)
				if "assignee_id" in json_data:
					newRow[12] = str(json_data["assignee_id"]) 
					
				# committers Ids
				newRow[13] = [str(commit["committer_id"]) for commit in json_data["linked_commits"]]

				# Edit Info Vector(s)
				newRow[14] = [editVector for editVector in json_data["edits_info"]]

				# Add the collected issue fields to the project array
				rows.append(newRow)
			
		# Write the project's issues' data to csv file
		with open(filename, 'w', newline='', encoding='UTF-8') as csvfile: 

			csvwriter = csv.writer(csvfile) 
			
			# write the first row with the fields
			csvwriter.writerow(fields)
			
			# write the issues data rows
			csvwriter.writerows(rows) 

# Function to create a directory for data after checking if it exists
def dirCreateCheck(dirName, functionToDo, *arg):

	# If directory doesn't exist, create new directory
	if not os.path.exists(dirName):
		
		os.makedirs(dirName)
		print("\nCreating new " + dirName + " directory...\n")
		
		# Call functionToDo with or without arg
		if arg:
			functionToDo(*arg)
		else:
			functionToDo()

	# If directory exists, ask the user if he wants to overwrite existing directory or continue
	else:

		user_input = input("The " + dirName + " directory already exists. Do you want to overwrite it? (y/n) ")
		if user_input == "y":
			shutil.rmtree(dirName)
			os.makedirs(dirName)
			print("Overwriting " + dirName + " directory...\n")
			functionToDo(arg)
		elif user_input == "n":
			print("Continuing without overwriting the existing " + dirName + " directory.\n")
		else:
			print("Invalid input. Continuing without overwriting the existing " + dirName + " directory.\n")

