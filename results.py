
def getProjectInfo(project, issuesNum, classesNum, casesDict, componUse, labelsUse):

    # Calculate InitIssuesNumber
    initIssuesNum = sum(casesDict.values())

    infoRow = {'Project': project,
               'InitialIssuesNumber': initIssuesNum,
               'IssuesNumber': issuesNum,
               'ClassesNumber': classesNum,
               'DevCase1': casesDict['Case 1'],
               'DevCase2': casesDict['Case 2'],
               'DevCase3.1': casesDict['Case 3.1'],
               'DevCase3.2': casesDict['Case 3.2'],
               'DevCase4': casesDict['Case 4'],
               'componentsUse': componUse,
               'labelsUse': labelsUse}
    
    return infoRow

def getMetricsResults(project, dfMetricName, metricIndex, resultsArray):

    # Get the values of the list for the given metric index
    metric_values = [lst[metricIndex] for lst in resultsArray]

    # Create a new row with the project name and the metric values
    new_row = {'Project': project}
    for i, col in enumerate(dfMetricName.columns[1:]):

        # Round every metric value to the third decimal place 
        new_row[col] = round(metric_values[i], 3)

    # Add the new row to the dataframe
    dfMetricName = dfMetricName.append(new_row, ignore_index=True)
    
    return dfMetricName