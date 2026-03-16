import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import mlflow
import dagshub

# Inicjalizacja połączenia z DagsHub i MLflow
dagshub.init(repo_owner='OlivierArthur', repo_name='BERT_porownanie_treningowych', mlflow=True)
mlflow.set_experiment("BERT_Spam_porown_zb_treng")

# Pobranie i rozpakowanie zbioru danych z Kaggle z nadpisaniem istniejących plików (-o)
os.system("kaggle datasets download -d nitishabharathi/email-spam-dataset --unzip -o")

datasety = [
    {"nazwa": "Exp_1_SpamAssassin", "csv_nazwa": "completeSpamAssassin.csv", "text": "Body", "label": "Label"},
    {"nazwa": "Exp_2_Enron",        "csv_nazwa": "enronSpamSubset.csv",      "text": "Body", "label": "Label"},
    {"nazwa": "Exp_3_LingSpam",     "csv_nazwa": "lingSpam.csv",             "text": "Body", "label": "Label"}
]

# Klasa konwertująca surowe tokeny i etykiety na tensory wymagane przez bibliotekę PyTorch
class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Inicjalizacja tokenizatora z pre-trenowanego modelu BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#Dla czytelności wykresów
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    acc = accuracy_score(labels, preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

for data in datasety:
    print(f"\n START EKSPERYMENTU: {data['nazwa']}")

    # Wczytanie danych (dodałem domyślny fallback w razie błędów kodowania)
    df = pd.read_csv(data["csv_nazwa"], encoding='utf-8', encoding_errors='replace')

    # Selekcja kolumn i usunięcie pustych wierszy (NaN)
    df = df[[data['text'], data['label']]].dropna()

    # Ograniczenie zbioru do max 2000 losowych próbek 
    df = df.sample(n=min(2000, len(df)), random_state=42)

    # Kodowanie etykiet tekstowych do wartości binarnych (0, 1)
    le = LabelEncoder()
    labels = le.fit_transform(df[data['label']])
    texts = df[data['text']].astype(str).tolist()

    # Podział na zbiór treningowy (80%) i walidacyjny (20%)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenizacja tekstów 
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    # Instancjacja datasetów w formacie zgodnym z PyTorchem
    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)

    # Rejestrowanie sesji treningowej w środowisku MLflow
    with mlflow.start_run(run_name=data['nazwa']):

        # Pobranie wag BERTa i dodanie warstwy klasyfikacyjnej na 2 klasy
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Konfiguracja hiperparametrów
        training_args = TrainingArguments(
            output_dir=f"./wyniki_{data['nazwa']}", 
            num_train_epochs=2,                      
            per_device_train_batch_size=16,          
            per_device_eval_batch_size=16,           
            eval_strategy="epoch",                   # Ewaluacja po każdej epoce
            learning_rate=2e-5,                      
            report_to="mlflow",                      
            logging_steps=10,                        
            weight_decay=0.01
        )

        # Obiekt Trainer 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics          # <--- NOWOŚĆ: Przekazanie funkcji z metrykami
        )

        print(f"Odpalamy trening na danych {data['csv_nazwa']}...")
        trainer.train()

        # Ręczne wywołanie ewaluacji na koniec treningu i wysłanie ostatecznych wyników do MLflow
        trainer.evaluate()

        # Zwolnienie zasobów karty graficznej
        del model
        del trainer
        import gc
        gc.collect() # Wymuszenie odśmiecacza pamięci
        torch.cuda.empty_cache()
        
        print(f" Zakończono: {data['nazwa']}")

print("\n KONIEC")
