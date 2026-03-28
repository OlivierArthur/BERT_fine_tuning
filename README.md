Konkretne parametry użyte podczas treningu można zobaczyć na https://huggingface.co/OliverArt5500/klasyfikatorspamu1 

W projekcie wykorzystano środowisko google colab z "!pip install evaluate".

Obraz na dockerze jest już stworzony i znajduje się na: https://hub.docker.com/repository/docker/oliverart5500/klasyfikatorspamu/general

Jak korzystać ( używając dockera ):
wpisać w terminalu
docker run -d -p 8080:8000 --name spam-api oliverart5500/klasyfikatorspamu








