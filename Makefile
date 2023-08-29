.PHONY: run-baseline run-mlp run-rnn run-lstm mlflow all

DOCKER_CMD = docker run --rm -v $(PWD):/assignment -p 8888:8888 spine
INSTALL_CMD = pip install --no-cache-dir -e /assignment
TUNING_CMD = python /assignment/scripts/hp_tuning.py --directory /assignment/data/processed/ --model_type

build:
	docker build -t spine .

run-baseline:
	@echo "Tuning BaselineModel"
	$(DOCKER_CMD) bash -c "$(INSTALL_CMD) && $(TUNING_CMD) BaselineModel"

run-mlp:
	@echo "Tuning MLP"
	$(DOCKER_CMD) bash -c "$(INSTALL_CMD) && $(TUNING_CMD) MLP"

run-rnn:
	@echo "Tuning RNN"
	$(DOCKER_CMD) bash -c "$(INSTALL_CMD) && $(TUNING_CMD) RNNModel"

run-lstm:
	@echo "Tuning LSTM"
	$(DOCKER_CMD) bash -c "$(INSTALL_CMD) && $(TUNING_CMD) LSTMModel"

mlflow:
	@echo "Running mlflow server"
	$(DOCKER_CMD) bash -c "mlflow server --backend-store-uri file:///assignment/mlruns -p 8888 --host 0.0.0.0"

jupyter:
	@echo "Running jupyter notebook"
	$(DOCKER_CMD) bash -c "$(INSTALL_CMD) && cd /assignment/notebooks && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"

run-all: build run-baseline run-mlp run-rnn run-lstm mlflow

