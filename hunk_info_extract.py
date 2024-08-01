from transformers import RobertaTokenizer
import difflib

# Define the CodeBERT model tokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')

# Function to check if a string is a programming comment
def isComment(text):

    # Check if line is a comment in Java, Python, Ruby, C, C++, JavaScript, Groovy, or shell script
    if text.startswith('//') or text.startswith('/*') or text.startswith('*') or text.startswith('#') or text.startswith(':') or text.startswith('"""') or text.startswith("'''") or text.startswith('=begin') or text.startswith('- FIX'):
        
        return True

    return False

# Function to edit a hunk and produce the desired sequences
def hunkEdit(text):

    # Split the input text into a list of strings, each representing a line of the text
    parts = text.split("\n")
    
    # Initialize empty lists to hold the old and new code
    old_code = []
    new_code = []
    
    # Initialize a variable to keep track of the current list being populated (old_code or new_code)
    current_list = None

    # Iterate over each line in the hunk parts
    for part in parts:

        # If the line is not a blank line or a line containing only whitespace and symbols (+ or -)
        if len(part.strip(' +-\n')) > 0:

            # If the line starts with a "-" symbol, add it to the old_code list keeping out the whitespace and the "-" symbol
            if part.startswith("-"):
                current_list = old_code
                current_list.append(part[1:].strip())
            
            # If the line starts with a "+" symbol, add it to the new_code list keeping out the whitespace and the "+" symbol
            elif part.startswith("+"):
                current_list = new_code
                current_list.append(part[1:].strip())

            # If the line doesn't start with either "+" or "-", and the current_list variable is not None, 
            # add the line to the current_list (either old_code or new_code)
            elif current_list is not None:
                current_list.append(part.strip())

    # Remove comment lines from the old and new code lists             
    old_code = [line for line in old_code if not isComment(line)]
    new_code = [line for line in new_code if not isComment(line)]

    # If both the old and new code lists are empty, return an empty list for all outputs (no useful edits)
    if len(old_code) == len(new_code) == 0:

        return [], [], []        
    
    # Join the old and new code lists into single strings
    merged_old_code = ' '.join(old_code)
    merged_new_code = ' '.join(new_code)

    # Tokenize the old and new code
    old_code_tokens = tokenizer.tokenize(merged_old_code)
    new_code_tokens = tokenizer.tokenize(merged_new_code)


    # If the new_code and old_code lists are too big (over 1000 tokens each), keep only the first 1000
    if len(old_code_tokens) > 1000 and len(new_code_tokens) > 1000:

        old_code_tokens = old_code_tokens[:1000]
        new_code_tokens = new_code_tokens[:1000]


    # Create a sequence matcher object to find the differences between the two tokenized code snippets
    match = difflib.SequenceMatcher(None, old_code_tokens, new_code_tokens)

    # Get the differences between the two tokenized code snippets
    differences = match.get_opcodes()

    # Initialize empty lists to store the modified old and new tokens
    modified_old_tokens = []
    modified_new_tokens = []

    # Loop over the differences and modify the token lists accordingly
    for tag, i1, i2, j1, j2 in differences:

        # If the edit operation is 'equal', add the tokens to both modified old and new code
        if tag == 'equal':
            modified_old_tokens += old_code_tokens[i1:i2]
            modified_new_tokens += new_code_tokens[j1:j2]

        # If the edit operation is 'replace', add the tokens to both modified old and new code with padding
        elif tag == 'replace':
            max_len = max(i2-i1, j2-j1)
            modified_old_tokens += old_code_tokens[i1:i2] + [tokenizer.pad_token] * (max_len-(i2-i1))
            modified_new_tokens += new_code_tokens[j1:j2] + [tokenizer.pad_token] * (max_len-(j2-j1))

        # If the edit operation is 'delete', add the tokens to modified old code with padding and add padding to modified new code
        elif tag == 'delete':
            if j2-j1 == 0:
                modified_old_tokens += old_code_tokens[i1:i2]
            else:
                modified_old_tokens += old_code_tokens[i1:i2] + [tokenizer.pad_token] * (j2-j1)
            modified_new_tokens += [tokenizer.pad_token] * (i2-i1 + j2-j1)
    
        # If the edit operation is 'insert', add the tokens to modified new code with padding and add padding to modified old code
        elif tag == 'insert':
            if i2-i1 == 0:
                modified_new_tokens += new_code_tokens[j1:j2]
            else:
                modified_new_tokens += new_code_tokens[j1:j2] + [tokenizer.pad_token] * (i2-i1)
            modified_old_tokens += [tokenizer.pad_token] * (i2-i1 + j2-j1)

    # Create a tag sequence indicating whether each token is unchanged ('='), added ('+'), deleted ('-'), or modified ('$')
    tag_sequence = []

    # Loop through the modified old and new code tokens and generate the tag sequence
    for old_token, new_token in zip(modified_old_tokens, modified_new_tokens):

        # If the token is a padding token, add a '+' tag to the tag sequence
        if old_token == tokenizer.pad_token:
            tag_sequence.append("+")

        # If the token is a padding token, add a '-' tag to the tag sequence
        elif new_token == tokenizer.pad_token:
            tag_sequence.append("-")

        # If the token is the same in old and new code, add a '=' tag to the tag sequence
        elif old_token == new_token:
            tag_sequence.append("=")
        
        # If the token is different in old and new code, add a '⇌' tag to the tag sequence
        else:
            tag_sequence.append("⇌")


    return modified_old_tokens, modified_new_tokens, tag_sequence


# Function that generates an edit representation vector that captures the semantic meaning of the code changes from a commit
def editRepresentation(commitObject):

    # Initialize empty lists to store old code sequence, new code sequence, and tag sequence
    oldCodeSequence = []
    newCodeSequence = []
    tagSequence = []

    # Iterate over code snippets in commitObject and extract hunk
    for code_snippet in commitObject["code_snippets"]:

        hunk = code_snippet["content"]

        # Generate modified old tokens, modified new tokens, and tag sequence using hunkEdit function
        modified_old_tokens, modified_new_tokens, tag_sequence = hunkEdit(hunk)
        
        # Append modified old tokens, modified new tokens, and tag sequence to corresponding lists
        oldCodeSequence = oldCodeSequence + modified_old_tokens
        newCodeSequence = newCodeSequence + modified_new_tokens
        tagSequence = tagSequence + tag_sequence

    # If there are no useful hunks (and the sequences all have length 0), return 0 for all
    if len(oldCodeSequence) == len(newCodeSequence) == len(tagSequence) == 0:

        return 0, 0, 0
    
    else:
        return oldCodeSequence, newCodeSequence, tagSequence
    
# Function that generates vector (list) with the percentages of each tag appearance and the length of tag sequence
def sequencePercentageVector(commit):

    # Get the tagSequence from the editRepresentation function
    _, _, sequence = editRepresentation(commit)

    # If the commit didn't have useful info and gave zero, return list of zeros
    if sequence == 0:
        return [0, 0, 0, 0, 0]

    # Length of tag sequence
    total_length = len(sequence)

    # Count the number of occurrences of each tag
    replace_count = sequence.count('⇌')
    add_count = sequence.count('+')
    remove_count = sequence.count('-')
    keep_count = sequence.count('=')

    # Calculate the percentage of each tag in the sequence
    replace_percentage = replace_count / total_length
    add_percentage = add_count / total_length
    remove_percentage = remove_count / total_length
    keep_percentage = keep_count / total_length

    # Create a vector of the percentages
    output_vector = [replace_percentage, add_percentage, remove_percentage, keep_percentage, total_length]

    return output_vector
