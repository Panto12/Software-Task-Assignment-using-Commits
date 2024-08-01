import os
import datetime
import pandas as pd
from db_connect import *
from preprocessing import preprocess_dataset
from data_train_test_evaluate import train_and_evaluate
from results import getMetricsResults

# Create directory for data and store data from the database to json files
dirCreateCheck("data", dataToJsonFiles)

# Create array to store projects and their number of issues
projectIssuesNumber = []

# Count issues found in each project and store number in array
for project in os.listdir("data"):
	lst = os.listdir("data/" + project)
	newRow = [0, 0]
	newRow[0] = project
	newRow[1] = len(lst)
	projectIssuesNumber.append(newRow)

# Find average issues number per project
numberOfProjects = len(projectIssuesNumber)
numberOfAllIssues = sum([row[1] for row in projectIssuesNumber])
avgIssuesPerProject = numberOfAllIssues / numberOfProjects
print("Average number of issues with commits per project: ", avgIssuesPerProject, "\n")

# Array with only the projects that have a higher number of issues than the average*1.2
projectsToUse = [row[0] for row in projectIssuesNumber if row[1] > avgIssuesPerProject*1.2]
print("Projects with more issues than average*1.2: \n", projectsToUse)

# Ask to preprocess the data gathered from the database. Needs to have been preprocessed at least once
user_input = input("\nDo you want to run the data preprocessing? (y/n) ")
if user_input == "y":
	print("Preprocessing project datasets...\n")
	for project in projectsToUse:
		preprocess_dataset(project)
elif user_input == "n":
	print("Continuing without preprocessing the data.\n")
else:
	print("Invalid input. Continuing without preprocessing the data.\n")

# Create directory for data in csv files and convert the data from the json files to csv files (one for each project)
dirCreateCheck("data_csv", dataToCsv, projectsToUse)

# Initialize empty dataframes for each metric
dfProjectInfo = pd.DataFrame(columns=['Project', 'InitialIssuesNumber', 'IssuesNumber', 'ClassesNumber', 'DevCase1', 'DevCase2', 'DevCase3.1', 'DevCase3.2', 'DevCase4', 'componentsUse', 'labelsUse'])
dfDevIssuesCounts = pd.DataFrame(columns=['Project', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'])
dfLossResults = pd.DataFrame(columns=['Project', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'])
dfAccuracyResults = pd.DataFrame(columns=['Project', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'])
dfPrecisionResults = pd.DataFrame(columns=['Project', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'])
dfRecallResults = pd.DataFrame(columns=['Project', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'])
dfF1_scoreResults = pd.DataFrame(columns=['Project', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'])

# Run training and evaluating model for every chosen project
for project in projectsToUse:
	
	print(project)
	projectInfo, devIssuesCounts, projectResults = train_and_evaluate(project)

	# Add the projectInfo row to the dfProjectInfo
	dfProjectInfo = dfProjectInfo.append(projectInfo, ignore_index=True)
		
	# Create a new row with the project name and devIssuesCounts
	devs_row_data = [project] + devIssuesCounts.tolist()

	# Pad the list with zeros until it has length 21
	devs_row_data += [0] * (21 - len(devs_row_data))
	
	# Append the row to the final dataframe
	dfDevIssuesCounts.loc[len(dfDevIssuesCounts)] = devs_row_data

	# Get the metrics results for each project and update the respective dataframe
	dfLossResults = getMetricsResults(project, dfLossResults, 0, projectResults)
	dfAccuracyResults = getMetricsResults(project, dfAccuracyResults, 1, projectResults)
	dfPrecisionResults = getMetricsResults(project, dfPrecisionResults, 2, projectResults)
	dfRecallResults = getMetricsResults(project, dfRecallResults, 3, projectResults)
	dfF1_scoreResults = getMetricsResults(project, dfF1_scoreResults, 4, projectResults)

# Create a directory for the results with the current datetime in the name
now = datetime.datetime.now().strftime('%m-%d_%H-%M')
resDirectory = f"Results_{now}"

# Create the "Results" directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Create the directory for the current results with the datetime in the name
resPath = os.path.join("Results", resDirectory)
os.makedirs(resPath)
   
# Write each dataframe to a separate Excel file inside the Results directory
dfProjectInfo.to_excel(os.path.join(resPath, 'Project_Info.xlsx'), index=False)
dfDevIssuesCounts.to_excel(os.path.join(resPath, 'Dev_Issues_Counts.xlsx'), index=False)
dfLossResults.to_excel(os.path.join(resPath, 'Loss_Results.xlsx'), index=False)
dfAccuracyResults.to_excel(os.path.join(resPath, 'Accuracy_Results.xlsx'), index=False)
dfPrecisionResults.to_excel(os.path.join(resPath, 'Precision_Results.xlsx'), index=False)
dfRecallResults.to_excel(os.path.join(resPath, 'Recall_Results.xlsx'), index=False)
dfF1_scoreResults.to_excel(os.path.join(resPath, 'F1_score_Results.xlsx'), index=False)

exit()
