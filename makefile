.PHONY: test_best_effort test_70_30 all clean killall collect results

test_best_effort:
	@sudo -E python3 -m pytest -k test_best_effort_all_no_control -s --html=best_effort.html --self-contained-html

test_70_30:
	@sudo -E python3 -m pytest -k test_70_30_hardcoded_split -s --html=70_30.html --self-contained-html

test_ai:
	@sudo -E python3 -m pytest -k test_ai -s --html=ai.html --self-contained-html

all:
	@sudo -E python3 -m pytest -s --html=test_report.html --self-contained-html

demo_dataset:
	@sudo -E python3 -m src.data_gen 30

dataset:
	@sudo -E python3 -m src.data_gen 600

model:
	@sudo -E python3 src/regr.py --data-csv data/bw_list_600.csv

results:
	python -m http.server

collect:
	@sudo -E python3 -m pytest -s --collect-only

killall:
	@sudo mn -c

clean: killall
	rm *.html -f
	rm *.log -f
	rm figs/* -f
	sudo ovs-vsctl -- --all destroy QoS -- --all destroy Queue
	rm data/*


