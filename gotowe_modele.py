
import os
from transformers import pipeline

df_pom = [
    {"nazwa": "Exp_1_SpamAssassin/checkpoint-530"},
    {"nazwa": "Exp_2_Enron/checkpoint-970"},
    {"nazwa": "Exp_3_LingSpam/checkpoint-260"}
]

# dać tu dwa zbiory testowe jeden z Subject:, jeden bez albo mieszane 
#os.system("kaggle datasets download -d nitishabharathi/email-spam-dataset --unzip -o")
#os.system("kaggle datasets download -d nitishabharathi/email-spam-dataset --unzip -o")

niezalezne_datasety_testowe = [
    {"nazwa": "", "csv_nazwa": "", "text": "", "label": ""},
    {"nazwa": "", "csv_nazwa": "", "text": "", "label": ""}
]

#zmienić nazwę ścieżki (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)

for nazwy in df_pom:
  klasyfikator = pipeline(
    task="text-classification", 
    model=f"./wyniki_{nazwy['nazwa']}", 
    tokenizer="bert-base-uncased",
    device=0 
  )

  #odkomentowac kiedy beda zbiory
  '''for test_data in niezalezne_datasety_testowe:
    print(f"\n Testowanie na zbiorze: {test_data['nazwa']}")
    df_test = pd.read_csv(test_data["csv_nazwa"], encoding='utf-8', encoding_errors='replace')
    df_test = df_test[[test_data['text'], test_data['label']]].dropna()
            
      
    y_true = le.fit_transform(df_test[test_data['label']])
    teksty = df_test[test_data['text']].astype(str).tolist()
            
    wyniki_surowe = klasyfikator(teksty, batch_size=32, truncation=True, max_length=128)
            
            # Wyciągnięcie przewidywanych etykiet (LABEL_0 -> 0, LABEL_1 -> 1)
    y_pred = [int(wynik['label'].replace('LABEL_', '')) for wynik in wyniki_surowe]
            
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary') 
    print(f" Wyniki to Accuracy: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")'''


  nowe_maile = [
    "URGENT: Your bank account has been locked. Click the link to verify your identity.",
    "Hey Arthur, let me know if you are available for a quick call tomorrow.",
    "Congratulations! You have won a free trip to Hawaii. Click the link to claim your prize.",
    "Subject: Work        Please be at work tomorrow ,. there is an important meeting",
    "QWE QWE         eeeeeeĘĘĘĘĘĘĘĘĘĘ AAAA LLVVSDWR 96486498 ***** "
    
  ]

  wyniki = klasyfikator(nowe_maile, truncation=True, max_length=128)
  print(wyniki)



