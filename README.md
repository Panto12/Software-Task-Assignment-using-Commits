# Improving Automated Software Task Assignment using Commits from Project Management Data

## Abstract 
Within the software development process, task assignment is a crucial aspect that seeks to assign these tasks for resolution to the most appropriate developer. Existing research in this area mostly concentrates on information gathered from software task reports, but there is still a lot of valuable information available in the related source code commits that has not been fully explored. 

This diploma thesis delves into the utilization of commit information to enhance the task assignment approach. By leveraging software task data and information from commits linked to the software tasks, we have developed a system aimed at augmenting the existing methodology. The primary objective was to evaluate the effectiveness of incorporating commit information compared to relying solely on software task reports. Through preprocessing techniques, we discovered that the commit fields contain valuable insights that can significantly contribute to the task assignment process. Towards this aim, we constructed a neural network model, specifically customized to our requirements. Preprocessing involved extracting and analyzing text fields from both software task reports and commits, as well as from the code content of the commits. Additionally, we encoded essential fields to extract meaningful features, thus enriching the dataset. The trained neural network model was then employed to classify software tasks, assigning them to the most suitable developer. 

To validate the efficacy of our approach, we divided the available data into project-specific subsets and performed training and testing procedures. By incorporating commit information into the software task assignment process, our system demonstrated the ability to achieve high precision and efficiency in assigning tasks to developers.



_Antonios Pontzo_

_Electrical and Computer Engineering School,_

_Aristotle University of Thessaloniki, Greece_

_June 2023_

## Dataset
The dataset used was from SmartSHARK database. We installed the SmartSHARK MongoDB Release 2.2 Small Version from [here](https://smartshark.github.io/dbreleases/).
1. Install SmartSHARK MongoDB.
2. Run [*db_hunk_update.py*](./db_hunk_update.py) to update the "hunk" collection in the database.

Note: the dataset is not included in the repo, due to size limitations.

## Code Run
In order to execute the full proccess described in this research, these steps should be followed:
1. Install the python dependencies as listed in [*requirements.txt*](./requirements.txt).
2. Execute [*main.py*](./main.py).
   * If no data directories (data and data_csv) are found in the project directory, they will be automatically created.
   * Else if the data directories are found, the user is asked to choose between overwriting them or not.
3. The projects to use are selected automatically and the whole method is executed for all of them.
   * If the user wishes to run the code for one project from the data, the user can use the `train_and_evaluate(project_name)` from [*data_train_test_evaluate.py*](./data_train_test_evaluate.py).
   * If the user wishes to filter the dataframe by keeping the issues of the top-k developers of the project, open [*data_train_test_evaluate.py*](./data_train_test_evaluate.py),
     uncomment the section with the comment: `# # Filter the df keeping the top-k devs` and specify the desired value for `k` in the function call.
