# Agent A: Auditor con LangGraph y RAG vectorial

Solucion profesional para auditar decisiones de un agente autonomo de seguros.
El agente usa un flujo orquestado con `LangGraph`, valida contratos con `Pydantic`,
consulta una base documental mediante embeddings reales de `sentence-transformers`
y puede incorporar un LLM local con Ollama para clasificar la accion tomada por Agent B.

## Arquitectura

```text
logs Agent B
  -> validacion Pydantic
  -> RAG vectorial sobre data/politicas
  -> analisis LLM estructurado
  -> evaluacion de reglas criticas
  -> indice de fidelidad analitica
  -> diagnostico y reporte JSON
```

El codigo principal esta concentrado en `src/auditor_agent.py` para que la solucion
sea facil de sustentar y no tenga archivos auxiliares innecesarios.

## Librerias usadas

- `langgraph`: orquestacion del flujo del agente auditor.
- `sentence-transformers`: generacion de embeddings semanticos.
- `numpy`: indice vectorial local y similitud coseno.
- `Ollama`: LLM local para clasificacion estructurada de la respuesta.
- `pydantic`: validacion de casos y reglas.
- `rich`: tabla ejecutiva en consola.

## Ejecucion

```bash
python3 src/auditor_agent.py audit
```

Modo demo rapido, sin depender de latencia del modelo local:

```bash
python3 src/auditor_agent.py audit --llm-provider deterministic
```

Formato exacto solicitado por el reto:

```bash
python3 src/auditor_agent.py audit --plain --llm-provider deterministic
```

El modelo usado es `sentence-transformers/all-MiniLM-L6-v2`. Debe estar descargado
en cache local para ejecutar sin red.
Para Ollama, el modelo por defecto es `llama3.2:3b` y puede cambiarse con `--ollama-model`.

## Pruebas

```bash
python3 -m unittest discover -s tests
```

## Docker

Construir imagen:

```bash
docker build -t agent-a-auditor .
```

Ejecutar demo estable:

```bash
docker run --rm agent-a-auditor
```

El contenedor ejecuta el modo deterministico para evitar depender de Ollama local
dentro de la imagen. Ollama puede usarse desde la maquina anfitriona ejecutando
directamente el comando Python con `--llm-provider ollama`.

## Base RAG

La base documental esta en `data/politicas/`:

- `poliza_auto.md`
- `poliza_vida.md`
- `sarlaft_aml.md`
- `siniestros_abuso.md`

Cada politica se vectoriza con embeddings. La recuperacion se hace por similitud
coseno contra la consulta formada con el contexto RAG y la respuesta de Agent B.

## Indice de Fidelidad Analitica

El indice va de 0 a 100 y combina:

- Alineacion de accion.
- Cumplimiento numerico.
- Controles criticos.
- Consistencia entre RAG vectorial y accion esperada.
- Analisis LLM estructurado como insumo de accion detectada.

Los pesos y umbrales viven en `reglas.json`.

## Resultado esperado

- Caso 1: conforme, aprueba una cobertura dentro del limite.
- Caso 2: conforme, escala una cuenta con sospecha de abuso.
- Caso 3: no conforme, aprueba USD 95,000 cuando el limite automatico es USD 80,000.
- Caso 4: bloqueo critico, ignora una alerta SARLAFT/AML.
