VENV := env

venv:
	python3 -m venv $(VENV) && source $(VENV)/bin/activate && pip3 install -r requirements.txt

run-multi:
	./$(VENV)/bin/python3 multi_agent.py