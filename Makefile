.PHONY: help backend-install backend-run backend-test frontend-install frontend-run frontend-build \
        seed-demo docker-up docker-down evaluate ml-train datasets

help:
	@echo "Common targets:"
	@echo "  make backend-install   Install backend Python dependencies"
	@echo "  make backend-run       Run the FastAPI dev server (http://localhost:8000)"
	@echo "  make backend-test      Run backend pytest suite"
	@echo "  make datasets          Download real datasets (CROHME, HASYv2, ASSISTments; ~1 GB)"
	@echo "  make ml-train          Retrain + evaluate all ML models (~8 min with real data, CPU)"
	@echo "  make seed-demo         Generate demo images + seed a demo student"
	@echo "  make frontend-install  Install frontend npm dependencies"
	@echo "  make frontend-run      Run the Vite dev server (http://localhost:5173)"
	@echo "  make frontend-build    Production build the frontend"
	@echo "  make docker-up         Build and start backend+frontend via docker-compose"
	@echo "  make docker-down       Stop docker-compose services"
	@echo "  make evaluate          Run the evaluation script against the demo manifest"

backend-install:
	cd backend && pip install --break-system-packages -r requirements.txt

backend-run:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

backend-test:
	cd backend && python -m pytest tests/ -v

datasets:
	cd backend && python scripts/download_datasets.py hasy crohme assistments

ml-train:
	cd backend && python -m app.ml.train

seed-demo:
	cd backend && python scripts/generate_demo_images.py && python scripts/seed_demo_data.py && python scripts/seed_demo_class.py

frontend-install:
	cd frontend && npm install

frontend-run:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

evaluate:
	cd evaluation && python benchmark.py --manifest manifests/demo_manifest.jsonl --providers ml demo
