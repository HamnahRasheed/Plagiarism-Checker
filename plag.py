import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#to access files from device
def load_files_from_folder(folder_path):
    files_dict = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                files_dict[filename] = content
                
    return files_dict

#Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered_tokens)

#Plagiarism Detection Functions 
def get_similarity(text1, text2):
    """Calculates cosine similarity between two texts."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def check_plagiarism(files_dict):
    """Compares files and returns similarity scores."""
    results = set()
    files = list(files_dict.keys())
    
    # Compare each file with every other file
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file1 = files[i]
            file2 = files[j]
            
            # Preprocess
            clean1 = preprocess_text(files_dict[file1])
            clean2 = preprocess_text(files_dict[file2])
            
            # Calculate similarity
            score = get_similarity(clean1, clean2)
            
            # Sort names to avoid duplicate (A,B) and (B,A) pairs
            pair = tuple(sorted((file1, file2)))
            results.add((pair[0], pair[1], score))
            
    return results

# Main Execution
if __name__ == "__main__":
    
    # Example student submissions
    folder_path = "student_submissions"
    student_files = load_files_from_folder(r"C:\Users\HAMNAHR\Desktop\stdSub")
    print("Checking for plagiarism...\n")
    plagiarism_results = check_plagiarism(student_files)
    
    # Print results
    for file1, file2, score in plagiarism_results:
        print(f"Similarity between {file1} and {file2}: {score:.2%}")
        if score > 0.5: # Threshold for suspected plagiarism
            print(f"  [!] High similarity detected.")