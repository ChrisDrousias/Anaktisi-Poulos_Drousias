import json
import re
import unicodedata
from collections import defaultdict

# Φόρτωση του ανεστραμμένου ευρετηρίου
with open('3. inverted_index.json', 'r', encoding='utf-8') as f:
    raw_inverted_index = json.load(f)

# Κανονικοποίηση κλειδιών ευρετηρίου
inverted_index = {}
for key, docs in raw_inverted_index.items():
    normalized_key = unicodedata.normalize('NFKD', key.strip().lower())
    inverted_index[normalized_key] = docs

# Συνάρτηση για τη διασύνδεση και επεξεργασία του ερωτήματος
def process_query(query):
    # Εάν το ερώτημα δεν περιέχει Boolean τελεστές, επιστρέφεται ως ενιαίος όρος
    if not any(op in query.lower() for op in ["and", "or", "not"]):
        normalized_query = unicodedata.normalize('NFKD', query.strip().lower())
        return [normalized_query]

    # Διαφορετικά, γίνεται κανονική ανάλυση σε tokens
    tokens = query.lower().strip().split()
    normalized_tokens = [unicodedata.normalize('NFKD', token.strip()) for token in tokens]
    return normalized_tokens

# Λειτουργίες Boolean

def boolean_and(set1, set2):
    return set1.intersection(set2)

def boolean_or(set1, set2):
    return set1.union(set2)

def boolean_not(set1, all_docs):
    return all_docs - set1

# Επεξεργασία Boolean ερωτήματος
def process_boolean_query(query, inverted_index):
    all_docs = set()

    # Επαλήθευση και προσαρμογή της δομής του ανεστραμμένου ευρετηρίου
    for key, docs in inverted_index.items():
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict):
                    # Προσθέτουμε μόνο μοναδικούς αναγνωριστικούς δείκτες (π.χ., "match")
                    if "match" in doc:
                        all_docs.add(doc["match"])
                else:
                    all_docs.add(doc)

    tokens = process_query(query)

    # Στοίβα για αξιολόγηση
    result_stack = []
    operators = []

    for token in tokens:
        if token in {"and", "or", "not"}:
            operators.append(token)
        else:
            # Ανάκτηση της λίστας εμφάνισης για τον όρο
            posting_list = set()
            normalized_token = unicodedata.normalize('NFKD', token.strip().lower())

            # Εύρεση του όρου στο κανονικοποιημένο ευρετήριο (Partial Match)
            for key in inverted_index.keys():
                if normalized_token in key:
                    for doc in inverted_index[key]:
                        if isinstance(doc, dict) and "match" in doc:
                            posting_list.add(doc["match"])

            result_stack.append(posting_list)

            # Αξιολόγηση αν υπάρχει ο τελεστής NOT
            while operators and operators[-1] == "not":
                operators.pop()
                operand = result_stack.pop()
                result_stack.append(boolean_not(operand, all_docs))

    # Αξιολόγηση τελεστών AND/OR στη στοίβα
    while operators:
        operator = operators.pop(0)  # FIFO
        operand1 = result_stack.pop(0)
        operand2 = result_stack.pop(0)

        if operator == "and":
            result_stack.append(boolean_and(operand1, operand2))
        elif operator == "or":
            result_stack.append(boolean_or(operand1, operand2))

    return result_stack[0] if result_stack else set()

# Διεπαφή χρήστη για αναζήτηση
def query_interface():
    print("Μηχανή Αναζήτησης Boolean Ερωτημάτων")
    print("Εισάγετε το ερώτημά σας (χρησιμοποιήστε AND, OR, NOT):")
    while True:
        query = input("Ερώτημα: ")
        if query.lower() == "exit":
            break
        results = process_boolean_query(query, inverted_index)
        print(f"Έγγραφα που ταιριάζουν στο ερώτημα: {results}")

if __name__ == "__main__":
    query_interface()
