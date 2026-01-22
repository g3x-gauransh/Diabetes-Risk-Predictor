# Makefile - Common project commands

.PHONY: help install clean train api ui test format lint docker

help:
	@echo "Diabetes Risk Predictor - Available Commands:"
	@echo ""
	@echo "  make install    - Install all dependencies"
	@echo "  make clean      - Remove cache and temporary files"
	@echo "  make train      - Train the model"
	@echo "  make api        - Start API server"
	@echo "  make ui         - Start Streamlit UI"
	@echo "  make test       - Run tests"
	@echo "  make format     - Format code with Black"
	@echo "  make lint       - Check code quality"
	@echo "  make docker     - Build Docker image"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	rm -f *.png *.log

train:
	python scripts/train_model.py

api:
	python api/app.py

ui:
	streamlit run ui/streamlit_app.py

test:
	pytest tests/ -v

format:
	black api/ config/ data/ models/ services/ utils/ scripts/ ui/

lint:
	flake8 api/ config/ data/ models/ services/ utils/ scripts/ --max-line-length=100

docker:
	docker build -t diabetes-predictor .