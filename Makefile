#Makefile
setup:
	python3 -m venv ~/.visiontransformer
	src ~/.visiontransformer/bin/activate 
	cd .visiontransformer
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
run:
	python src/test.py
all: install run
