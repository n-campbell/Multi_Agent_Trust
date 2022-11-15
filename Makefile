VENV := env

venv:
	python3 -m venv $(VENV) && source $(VENV)/bin/activate && pip3 install -r requirements.txt

run-circle: 
	./$(VENV)/bin/python3 circle_obs.py

run-unicycle:
	./$(VENV)/bin/python3 unicycle_main.py

run-sphere:
	./$(VENV)/bin/python3 sphere_obs.py
run-multi:
	./$(VENV)/bin/python3 multi_agent.py