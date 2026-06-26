# Agent A: auditor de decisiones de IA en seguros

Este proyecto implementa un agente auditor para revisar decisiones generadas por
un agente autonomo de seguros. La solucion combina recuperacion semantica de
politicas, validacion de reglas de negocio y una salida trazable para explicar
por que una transaccion puede continuar, escalarse o bloquearse.

## Flujo de trabajo

```text
logs de Agent B
  -> validacion de estructura
  -> recuperacion RAG sobre politicas internas
  -> clasificacion de la accion tomada
  -> controles de negocio y riesgo
  -> indice de fidelidad analitica
  -> diagnostico final
```

El codigo principal esta en `src/auditor_agent.py`. Las reglas de control viven
en `reglas.json` y las politicas usadas por el RAG estan en `data/politicas/`.

## Componentes

- `LangGraph`: organiza el flujo del agente.
- `sentence-transformers`: genera embeddings semanticos.
- `NumPy`: calcula similitud coseno sobre los vectores.
- `Pydantic`: valida datos de entrada, reglas y salida estructurada.
- `Ollama`: permite usar un LLM local para clasificar la respuesta de Agent B.
- `Rich`: muestra una tabla ejecutiva en consola.

## Ejecucion

Comandos:

```bash
python3 src/auditor_agent.py audit --plain --llm-provider deterministic
```

Este comando imprime la salida con el formato solicitado por el reto.

Para ver una tabla resumida en consola:

```bash
python3 src/auditor_agent.py audit --llm-provider deterministic
```

Si Ollama esta levantado localmente:

```bash
python3 src/auditor_agent.py audit --plain --llm-provider ollama
```

El modelo de embeddings es `sentence-transformers/all-MiniLM-L6-v2`. Para Ollama,
el modelo por defecto es `llama3.2:3b` y puede cambiarse con `--ollama-model`.

## Docker

```bash
docker build -t agent-a-auditor .
docker run --rm agent-a-auditor
```

La imagen usa el modo deterministico para mantener una ejecucion reproducible.
Durante el build se descarga el modelo de embeddings usado por el RAG.
Ollama se puede usar por fuera del contenedor cuando se quiera probar el flujo
con LLM local.

## Pruebas

```bash
python3 -m unittest discover -s tests
```

## Base de conocimiento

```text
data/politicas/
  poliza_auto.md
  poliza_vida.md
  sarlaft_aml.md
  siniestros_abuso.md
```

Cada politica se convierte en embedding. El caso auditado tambien se vectoriza y
se compara contra esas politicas mediante similitud coseno.

## Indice de Fidelidad Analitica

El indice va de 0 a 100 y combina:

- Alineacion de accion.
- Cumplimiento numerico.
- Controles criticos.
- Consistencia entre RAG vectorial y accion esperada.
- Clasificacion estructurada de la respuesta de Agent B.

Los pesos y umbrales se configuran en `reglas.json`.

## Casos cubiertos

- Caso 1: aprobacion conforme, monto dentro del limite de cobertura.
- Caso 2: escalamiento conforme por sospecha de abuso.
- Caso 3: no conforme, aprobacion por encima del limite automatico.
- Caso 4: bloqueo critico, alerta SARLAFT/AML ignorada.

## Escalamiento cloud

Para produccion, el agente auditor se puede desplegar como un servicio
independiente entre Agent B y los sistemas transaccionales de la aseguradora.

Arquitectura sugerida:

```text
Canales digitales
  -> Agent B
  -> Agent A Auditor
  -> decision: continuar, escalar o bloquear
  -> sistemas core / cola de analistas
```

Componentes recomendados:

- API Gateway para exponer el servicio de auditoria.
- Contenedor o funcion serverless para ejecutar Agent A.
- Vector store administrado para politicas y clausulados.
- Repositorio versionado para `reglas.json`.
- Event bus para auditoria asincronica de alto volumen.
- Data lake para trazabilidad historica.
- Monitoreo de latencia, errores, drift semantico y casos no conformes.
- Tablero operativo para Riesgos, Tecnologia y Cumplimiento.

El flujo sincrono debe reservarse para decisiones de alto impacto, como pagos,
emisiones, listas restrictivas o desviaciones de limite. Para monitoreo masivo,
el mismo agente puede ejecutarse de forma asincronica sobre eventos, sin afectar
la experiencia del cliente.
