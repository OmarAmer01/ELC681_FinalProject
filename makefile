.PHONY: test clean killall collect all

test_best_effort:
	@sudo -E python3 -m pytest -k test_best_effort_all_no_control -s --html=report.html --self-contained-html

make test_70_30:
	@sudo -E python3 -m pytest -k test_70_30_hardcoded_split -s --html=report.html --self-contained-html

all: test_best_effort

results:
	python -m http.server

collect:
	@sudo -E python3 -m pytest -s --collect-only

killall:
	@sudo mn -c

clean: killall
	rm report.html -f
	rm figs/* -f
	sudo ovs-vsctl -- --all destroy QoS -- --all destroy Queue

