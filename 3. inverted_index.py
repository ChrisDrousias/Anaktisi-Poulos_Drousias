import json
from collections import defaultdict

# Φόρτωση του JSON αρχείου
with open("E:\\.uni\\AnaktisiPliroforias\\football\\goalscorers_processed.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Δημιουργία ανεστραμμένου πίνακα
inverted_index = defaultdict(list)

for idx, record in enumerate(data):
    scorer = record["scorer"]
    team = record["team"]
    match_key = f"{record['date']} - {record['home_team']} vs {record['away_team']}"

    # Προσθήκη στο ανεστραμμένο ευρετήριο
    inverted_index[scorer].append({
        "team": team,
        "match": match_key,
        "minute": record["minute"],
        "own_goal": record["own_goal"],
        "penalty": record["penalty"]
    })

# Αποθήκευση του ανεστραμμένου πίνακα σε αρχείο
with open("inverted_index.json", "w", encoding="utf-8") as outfile:
    json.dump(inverted_index, outfile, ensure_ascii=False, indent=4)

print("Ο ανεστραμμένος πίνακας δημιουργήθηκε και αποθηκεύτηκε στο inverted_index.json.")
