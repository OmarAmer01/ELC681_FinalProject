.PHONY: test clean killall collect

test:
	@sudo -E python3 -m pytest -s --html=report.html --self-contained-html

results:
	python -m http.server

collect:
	@sudo -E python3 -m pytest -s --collect-only

killall:
	@sudo mn -c

clean: killall
	rm report.html -f

