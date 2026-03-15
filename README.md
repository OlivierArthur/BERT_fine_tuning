Uwaga: kod jest jedynie poglądowy, nie ma wersji użytej do pierwszego eksperymentu w repozytorium - ale jest opisane co trzeba zmienić żeby go odtworzyć.

Porównanie trzech różnych zbiorów danych z emailami, dwa z bardziej realistyczną proporcją spamu do prawdziwych wiadomości, jeden 50/50.
W danych pojawiają się "Subject: " (tylko w jednym zbiorze) i ciągi mailów, w pierwszej iteracji eksperymentu chcemy je zostawić i zobaczyć co się dzieje ( czy zepsują coś? )

W pierwszej iteracji porównania training setów w TrainingArguments nie było weight decay, były 3 epoki i learning rate 3e-5. Na wykresach można było zaobserwować overtraining w zbiorach treningowych enron i lingspam. W przypadku SpamAssassin trenowanie przez 3 epoki z większym learning rate było efektywne, co można zobaczyć na wykresach w folderach.
Postać TrainingArguments w eksperymencie1: 

training_args = TrainingArguments(
            output_dir=f"./wyniki_{data['nazwa']}", #Ścieżka dla checkpointów
            num_train_epochs=3,                     #Liczba epok zmieniona na 2 zamiast 3 przez overfitting
            per_device_train_batch_size=16,         #Wielkość paczki treningowej
            per_device_eval_batch_size=16,          #Wielkość paczki walidacyjnej
            eval_strategy="epoch",                  #Walidacja po każdej epoce
            learning_rate=3e-5,                     #Współczynnik uczenia
            report_to="mlflow",                     #Wymuszenie wysyłania logów do DagsHuba/MLflow
            logging_steps=10,                        #Częstotliwość raportowania metryk
        )

Postać TrainingArguments w eksperymencie2: 
 training_args = TrainingArguments(
            output_dir=f"./wyniki_{data['nazwa']}", #Ścieżka dla checkpointów
            num_train_epochs=2,                     #Liczba epok zmieniona na 2 zamiast 3 przez overfitting
            per_device_train_batch_size=16,         #Wielkość paczki treningowej
            per_device_eval_batch_size=16,          #Wielkość paczki walidacyjnej
            eval_strategy="epoch",                  #Walidacja po każdej epoce
            learning_rate=2e-5,                     #Współczynnik uczenia
            report_to="mlflow",                     #Wymuszenie wysyłania logów do DagsHuba/MLflow
            logging_steps=10,                        #Częstotliwość raportowania metryk
            weight_decay=0.01
        )

