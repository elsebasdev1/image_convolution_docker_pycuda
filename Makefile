# Uso: make up   -> construye, arranca el contenedor y deja la web en http://localhost:8000
.DEFAULT_GOAL := help
export GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo desconocido)
URL ?= http://localhost:8000

.PHONY: help up down restart logs status gpu test shell bench medir

help:  ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

up:  ## Construye la imagen, arranca el contenedor y espera a que la GPU esté lista
	docker compose up -d --build
	@echo "Esperando al servicio..."
	@for i in $$(seq 1 60); do \
	  curl -sf $(URL)/api/health >/dev/null 2>&1 && break; sleep 1; \
	done
	@curl -sf $(URL)/api/health | python3 -c "import sys,json; g=json.load(sys.stdin)['gpu']; print('GPU:', g.get('nombre') or g.get('error'))" \
	  || (echo "El servicio no respondio: make logs" && exit 1)
	@echo "Web lista en $(URL)   (docs de la API: $(URL)/docs)"

down:  ## Detiene el contenedor
	docker compose down

restart: down up  ## Reinicia el servicio

logs:  ## Sigue los registros del contenedor
	docker compose logs -f convolucion

status:  ## Estado del contenedor y de la GPU vista por el servicio
	@docker compose ps
	@curl -s $(URL)/api/health | python3 -m json.tool | head -20

gpu:  ## Verifica la GPU dentro del contenedor (nvidia-smi, nvcc, PyCUDA)
	scripts/verificar_gpu_contenedor.sh

test:  ## Ejecuta las pruebas dentro del contenedor
	docker run --rm --gpus all convolucion-gpu:1.0 python -m pytest -q

shell:  ## Abre una shell en el contenedor en marcha
	docker compose exec convolucion bash

bench:  ## Mediciones contra el servicio en marcha y resumen (make up antes)
	uv run python -m bench.cliente --modo servicio_contenedor --url $(URL)
	uv run python -m bench.resumir

medir:  ## Protocolo completo (host, contenedor, servicio, concurrencia, validación)
	scripts/medir_todo.sh
