
import os
import re
import json

# Function to process a file and extract unique words and tokens
def processFile(file_path, error_log_path):
    word_dict = {
        "file_name": file_path,  # Store the file name
        "words": set(),         # Use a set to store unique words
        "tokens": set()         # Use a set to store unique tokens
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                words = [re.sub(r'[^\u0980-\u09FFa-zA-Z0-9]', '', word) for word in line.strip().split()]
                word_dict["words"].update(words)
                for word in words:
                    word =  re.sub(r'[^\w\s]', '', word)
                    tokens = re.findall(r'[\u0980-\u09FF]', word)
                    word_dict["tokens"].update(tokens)  # Add unique tokens to the set
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        with open(error_log_path, 'a', encoding='utf-8') as error_log:
            error_log.write(f"Error processing file {file_path}: {e}\n")

    return word_dict


# Function to process all files in a directory
def processDirectory(directory_path, error_log_path):
    all_data = []  # List to store data for all files
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return all_data

    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".txt")]
    if not file_paths:
        print("No .txt files found in the directory.")
        return all_data

    for file_path in file_paths:
        file_data = processFile(file_path, error_log_path)
        if file_data["words"] or file_data["tokens"]:
            all_data.append(file_data)

    return all_data


# Function to load dictionary words from a file
def loadWordDictionary(dictionary_file_path, error_log_path):
    word_set = set()
    try:
        with open(dictionary_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()
                if word:
                    word_set.add(word)  # Add to set to avoid duplicates
    except Exception as e:
        with open(error_log_path, 'a', encoding='utf-8') as error_log:
            error_log.write(f"Error loading dictionary file {dictionary_file_path}: {e}\n")
    return word_set


# Function to compare words in the dictionary
def compare_words(word_set, dictionary_set):
    return {word for word in word_set if word in dictionary_set}

def write_file_data_to_output(output_file_path, all_file_data):
    try:
        if not all_file_data:
            print("No data to write to the output file.")
            return

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for file_data in all_file_data:
                file_data['words'] = list(file_data['words'])
                file_data['tokens'] = list(file_data['tokens'])
                
                # Write each file_data object as a single-line JSON
                output_file.write(json.dumps(file_data, ensure_ascii=False) + '\n')

        print(f"\nProcessed data has been successfully written to: {output_file_path}")
    except IOError as e:
        print(f"IO error while writing to output file {output_file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error while writing to output file: {e}")

if __name__ == "__main__":
    # Paths
    data_directory = "./data"
    dictionary_file_path = "./dictionary.txt"
    error_log_path = "./error_log.txt"
    output_file_path = "./processed_data.txt"  # Output file for processed data


    # Clear the error log file before starting
    open(error_log_path, 'w').close()

    # Step 1: Process all files in the data directory
    all_file_data = processDirectory(data_directory, error_log_path)

    if all_file_data:
        # Write processed data to output file
        write_file_data_to_output(output_file_path, all_file_data)

    # Step 2: Load dictionary words
    word_dictionary = loadWordDictionary(dictionary_file_path, error_log_path)
    print("\nLoaded Dictionary Words:", word_dictionary)

    # Step 3: Compare words in each file with the dictionary
    for file_data in all_file_data:
        matched_words = compare_words(file_data["words"], word_dictionary)
        print(f"\nMatched Words in File {file_data['file_name']}: {matched_words}")

    if os.path.getsize(error_log_path) > 0:
        print(f"\nErrors occurred during processing. Check the error log: {error_log_path}")
print("END OF FILE")