.PHONY: test_best_effort test_70_30 all clean killall collect results

test_best_effort:
	@sudo -E python3 -m pytest -k test_best_effort_all_no_control -s --html=report.html --self-contained-html

test_70_30:
	@sudo -E python3 -m pytest -k test_70_30_hardcoded_split -s --html=report.html --self-contained-html

all:
	@sudo -E python3 -m pytest -s --html=report.html --self-contained-html

demo_dataset:
	@sudo -E python3 src/data_gen.py 30

dataset:
	@sudo -E python3 src/data_gen.py 600

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
	rm data/*

