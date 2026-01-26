.PHONY: test

test:
	@sudo -E python3 -m pytest -s
# 	@sudo PYTHONPATH=$(PWD) python3 -m pytest -s
