import json
import unicodedata
from collections import defaultdict, Counter
import math

# Φόρτωση του ανεστραμμένου ευρετηρίου
with open('3. inverted_index.json', 'r', encoding='utf-8') as f:
    raw_inverted_index = json.load(f)

# Κανονικοποίηση κλειδιών ευρετηρίου και κατακερματισμός σε λέξεις
inverted_index = defaultdict(list)
word_to_docs = defaultdict(set)

for key, docs in raw_inverted_index.items():
    normalized_key = unicodedata.normalize('NFKD', key.strip().lower())
    inverted_index[normalized_key] = docs

    # Διαχωρισμός των κλειδιών σε λέξεις
    for word in normalized_key.split():
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict) and "match" in doc:
                    word_to_docs[word].add((doc["match"], doc.get("minute", "N/A")))
                elif not isinstance(doc, dict):
                    word_to_docs[word].add((doc, "N/A"))
        elif isinstance(docs, dict) and "match" in docs:
            word_to_docs[word].add((docs["match"], docs.get("minute", "N/A")))
        elif not isinstance(docs, dict):
            word_to_docs[word].add((docs, "N/A"))

# Υπολογισμός TF-IDF

def compute_tf_idf(query, inverted_index, word_to_docs):
    """Υπολογισμός TF-IDF για την κατάταξη αποτελεσμάτων."""
    # Υπολογισμός συνολικού αριθμού εγγράφων
    total_docs = sum(len(docs) if isinstance(docs, list) else 1 for docs in inverted_index.values())

    # Καταγραφή συχνότητας όρων στο query
    normalized_query = unicodedata.normalize('NFKD', query.strip().lower())

    # Ελέγχει αν το query περιέχει ακέραια φράση
    phrase_scores = defaultdict(float)
    if normalized_query in inverted_index:
        doc_list = inverted_index[normalized_query]
        doc_frequency = len(doc_list)

        # Υπολογισμός IDF για ολόκληρη τη φράση
        idf = math.log(total_docs / (1 + doc_frequency))

        for doc in doc_list:
            if isinstance(doc, dict) and "match" in doc:
                phrase_scores[(doc["match"], doc.get("minute", "N/A"))] += idf

    # Υπολογισμός TF-IDF για μεμονωμένα tokens
    scores = defaultdict(float)
    for term in normalized_query.split():
        if term in word_to_docs:
            doc_list = word_to_docs[term]
            doc_frequency = len(doc_list)

            # Υπολογισμός IDF
            idf = math.log(total_docs / (1 + doc_frequency))

            for doc_id, minute in doc_list:
                # Υπολογισμός TF
                scores[(doc_id, minute)] += idf

    # Συνδυασμός φράσης και μεμονωμένων tokens, δίνοντας προτεραιότητα στις πλήρεις φράσεις
    for key, value in phrase_scores.items():
        scores[key] += value * 2  # Υψηλότερη βαρύτητα στις πλήρεις φράσεις

    # Ταξινόμηση εγγράφων
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results

# Διεπαφή χρήστη για αναζήτηση

def query_interface():
    print("Μηχανή Αναζήτησης Boolean Ερωτημάτων και Κατάταξης (TF-IDF)")
    print("Εισάγετε το ερώτημά σας (χρησιμοποιήστε AND, OR, NOT ή απλά λέξεις):")
    while True:
        query = input("Ερώτημα: ")
        if query.lower() == "exit":
            break

        # Υπολογισμός κατάταξης με TF-IDF
        ranked_results = compute_tf_idf(query, inverted_index, word_to_docs)

        # Εμφάνιση αποτελεσμάτων στον χρήστη
        if ranked_results:
            print("Αποτελέσματα κατάταξης:")
            for rank, ((doc_id, minute), score) in enumerate(ranked_results, start=1):
                print(f"{rank}. {doc_id} (Λεπτό: {minute}) (Βαθμολογία: {score:.4f})")
        else:
            print("Δεν βρέθηκαν αποτελέσματα για το ερώτημά σας.")

if __name__ == "__main__":
    query_interface()
