import json
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Κατεβάζουμε τα απαραίτητα πακέτα NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Φόρτωση JSON αρχείου
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Συνάρτηση προεπεξεργασίας κειμένου
def preprocess_text(text):
    if not isinstance(text, str):
        return text  # Επιστρέφει το αρχικό αν δεν είναι string

    # Αφαίρεση ειδικών χαρακτήρων (εκτός κενών και παύλας)
    text = re.sub(r'[^\w\s-]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Αφαίρεση stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization (Διατήρηση κεφαλαίων)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)  # Επιστροφή επεξεργασμένου κειμένου ως string

# Επεξεργασία όλων των πεδίων του dataset
def preprocess_dataset(dataset):
    for entry in dataset:
        for key in entry:
            if isinstance(entry[key], str):  # Επεξεργασία μόνο για string
                entry[key] = preprocess_text(entry[key])
            elif isinstance(entry[key], bool):  # Διατήρηση των boolean τιμών
                entry[key] = entry[key]
    return dataset

# Αποθήκευση JSON σε μορφή μίας γραμμής ανά εγγραφή
# def save_json_inline(data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         for entry in data:
#             # Μετατροπή της εγγραφής σε JSON string μίας γραμμής
#             json_line = json.dumps(entry, ensure_ascii=False)
#             file.write(json_line + '\n')  # Προσθέτουμε νέα γραμμή μετά από κάθε εγγραφή

# Αποθήκευση JSON ως έγκυρο JSON αρχείο (λίστα εγγραφών)
def save_json_inline(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        # Αποθηκεύει όλες τις εγγραφές ως μία λίστα
        json.dump(data, file, ensure_ascii=False, indent=4)  # Χρήση indent για πιο αναγνώσιμο JSON

# Κύριο πρόγραμμα
input_file = 'E:/.uni/AnaktisiPliroforias/football/goalscorers.json'  # Διαδρομή εισόδου
output_file = 'E:/.uni/AnaktisiPliroforias/football/goalscorers_processed.json'  # Διαδρομή εξόδου

# Φόρτωση δεδομένων
data = load_json(input_file)

# Επεξεργασία δεδομένων
processed_data = preprocess_dataset(data)

# Αποθήκευση δεδομένων σε γραμμική μορφή
save_json_inline(processed_data, output_file)

print(f"Η αποθήκευση ολοκληρώθηκε. Το αρχείο αποθηκεύτηκε στη διαδρομή: {output_file}")
