.PHONY: help install install-dev install-prod clean test test-cov lint format check-quality docker-build docker-run docker-stop docker-clean docker-logs setup-db migrate seed-data run run-dev run-prod health-check logs monitor setup-env setup-precommit

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
DOCKER := docker
DOCKER_COMPOSE := docker-compose
APP_NAME := meet-mind
PYTHON_VERSION := 3.11

# Default target
help: ## Show this help message
	@echo "MeetMind - Advanced RAG-Powered Knowledge Assistant"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt || echo "No dev requirements found"

install-prod: ## Install production dependencies with optimizations
	$(PIP) install --no-cache-dir -r requirements.txt

# Environment setup
setup-env: ## Setup environment variables
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env || echo "No .env.example found, creating basic .env"; \
		echo "ENVIRONMENT=development" >> .env; \
		echo "DEBUG=true" >> .env; \
		echo "SECRET_KEY=$(shell python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env; \
		echo "DATABASE_URL=sqlite:///./meetmind.db" >> .env; \
		echo "QDRANT_HOST=localhost" >> .env; \
		echo "QDRANT_PORT=6333" >> .env; \
		echo "REDIS_URL=redis://localhost:6379" >> .env; \
		echo ".env file created. Please review and update as needed."; \
	else \
		echo ".env file already exists."; \
	fi

setup-precommit: ## Setup pre-commit hooks
	$(PIP) install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg

# Code quality
lint: ## Run linting checks
	@echo "Running flake8..."
	$(FLAKE8) app/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Running mypy..."
	$(MYPY) app/ --ignore-missing-imports --no-strict-optional

format: ## Format code with black and isort
	@echo "Formatting with black..."
	$(BLACK) app/ tests/
	@echo "Sorting imports with isort..."
	$(ISORT) app/ tests/

check-quality: ## Run all code quality checks
	@echo "Running code quality checks..."
	@make format
	@make lint
	@echo "Code quality checks completed!"

# Testing
test: ## Run tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=app --cov-report=html --cov-report=term-missing

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v

test-performance: ## Run performance tests
	$(PYTEST) tests/performance/ -v

test-all: ## Run all tests with coverage
	@make test-cov
	@make test-integration
	@make test-performance

# Database operations
setup-db: ## Setup database and run migrations
	@echo "Setting up database..."
	alembic upgrade head

migrate: ## Create and run new migration
	@read -p "Enter migration message: " message; \
	alembic revision --autogenerate -m "$$message"
	alembic upgrade head

seed-data: ## Seed database with sample data
	@echo "Seeding database with sample data..."
	$(PYTHON) scripts/seed_data.py

# Docker operations
docker-build: ## Build Docker image
	$(DOCKER) build -t $(APP_NAME):latest -f docker/Dockerfile .

docker-run: ## Run application with Docker Compose
	$(DOCKER_COMPOSE) up -d

docker-stop: ## Stop Docker Compose services
	$(DOCKER_COMPOSE) down

docker-clean: ## Clean up Docker resources
	$(DOCKER_COMPOSE) down -v --remove-orphans
	$(DOCKER) system prune -f

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-restart: ## Restart Docker services
	$(DOCKER_COMPOSE) restart

# Application operations
run: ## Run application in development mode
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-dev: ## Run application in development mode with debug
	ENVIRONMENT=development DEBUG=true uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run application in production mode
	ENVIRONMENT=production DEBUG=false uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Monitoring and health
health-check: ## Check application health
	@echo "Checking application health..."
	curl -f http://localhost:8000/health || echo "Application is not responding"

logs: ## View application logs
	tail -f logs/app.log

monitor: ## Monitor application metrics
	@echo "Application metrics available at:"
	@echo "- Health: http://localhost:8000/health"
	@echo "- Metrics: http://localhost:8000/metrics"
	@echo "- Prometheus: http://localhost:9090"
	@echo "- Grafana: http://localhost:3000 (admin/admin)"

# Development utilities
clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

reset: ## Reset development environment
	@echo "Resetting development environment..."
	@make docker-clean
	@make clean
	@rm -f .env
	@rm -f *.db
	@rm -f *.sqlite
	@echo "Development environment reset complete!"

# Performance and profiling
profile: ## Run performance profiling
	$(PYTHON) -m cProfile -o profile.stats app/main.py

profile-view: ## View profiling results
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

benchmark: ## Run performance benchmarks
	$(PYTHON) scripts/benchmark.py

# Security checks
security-check: ## Run security checks
	@echo "Running security checks..."
	bandit -r app/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "Security check reports generated: bandit-report.json, safety-report.json"

# Documentation
docs: ## Generate API documentation
	@echo "API documentation available at:"
	@echo "- Swagger UI: http://localhost:8000/docs"
	@echo "- ReDoc: http://localhost:8000/redoc"
	@echo "- OpenAPI JSON: http://localhost:8000/openapi.json"

# Deployment
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	# Add your staging deployment commands here

deploy-production: ## Deploy to production environment
	@echo "Deploying to production..."
	# Add your production deployment commands here

# Quick start
quick-start: ## Quick start for development
	@echo "Setting up MeetMind development environment..."
	@make setup-env
	@make install-dev
	@make setup-precommit
	@make docker-run
	@echo "Waiting for services to start..."
	@sleep 10
	@make setup-db
	@make health-check
	@echo "MeetMind is ready! Run 'make run' to start the application."

# Development workflow
dev-workflow: ## Complete development workflow
	@echo "Starting development workflow..."
	@make check-quality
	@make test
	@make docker-build
	@make docker-run
	@echo "Development workflow completed!"

# Utility functions
check-deps: ## Check if all dependencies are installed
	@echo "Checking dependencies..."
	@$(PYTHON) -c "import fastapi, sentence_transformers, qdrant_client, openai; print('All dependencies are installed!')" || echo "Some dependencies are missing. Run 'make install' to install them."

check-ports: ## Check if required ports are available
	@echo "Checking port availability..."
	@netstat -tuln | grep -E ":(8000|5432|6333|6379|9090|3000)" || echo "All required ports are available!"

# Database backup and restore
backup-db: ## Backup database
	@echo "Backing up database..."
	@if [ -f meetmind.db ]; then \
		cp meetmind.db "backup_meetmind_$$(date +%Y%m%d_%H%M%S).db"; \
		echo "Database backed up successfully!"; \
	else \
		echo "No database file found to backup."; \
	fi

restore-db: ## Restore database from backup
	@echo "Available backups:"
	@ls -la backup_meetmind_*.db 2>/dev/null || echo "No backups found"
	@if [ -f meetmind.db ]; then \
		echo "Current database will be replaced. Continue? (y/N)"; \
		read -r response; \
		if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
			echo "Enter backup filename:"; \
			read -r backup_file; \
			if [ -f "$$backup_file" ]; then \
				cp "$$backup_file" meetmind.db; \
				echo "Database restored successfully!"; \
			else \
				echo "Backup file not found!"; \
			fi; \
		else \
			echo "Restore cancelled."; \
		fi; \
	else \
		echo "No current database to replace."; \
	fi
