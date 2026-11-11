.PHONY: run-docker-compose

run-docker-compose:
	docker compose up -d --build
	@echo "FastAPI Docs:     http://localhost:8000/docs"
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"
	@echo "Qdrant API:       http://localhost:6333"
