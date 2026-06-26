# Guion ejecutivo - Comite de Riesgos y Tecnologia

## Slide 1 - La decision autonoma necesita control preventivo

Agent B mejora velocidad, pero tambien puede aprobar pagos, emisiones o beneficiarios fuera del apetito de riesgo.
Agent A actua como segunda linea automatizada antes de que la decision impacte produccion.

## Slide 2 - Arquitectura limpia y defendible

El flujo usa LangGraph para orquestar, Pydantic para validar contratos, RAG vectorial con
`sentence-transformers/all-MiniLM-L6-v2` para consultar politicas, Ollama local para clasificar
la accion de Agent B en JSON estructurado y `reglas.json` para aplicar controles duros.

## Slide 3 - Evidencia de los 4 casos

Caso 1 y 2 pasan porque respetan limite o escalan correctamente.
Caso 3 falla por superar el limite de emision automatica.
Caso 4 es critico porque una alerta SARLAFT/AML exige bloqueo inmediato.

## Slide 4 - Lectura de riesgo

La solucion separa decisiones operables de desviaciones que requieren analista o Cumplimiento.
El valor no es solo clasificar, sino dejar trazabilidad: fuentes RAG, controles activados, accion esperada y diagnostico.

## Slide 5 - Escalamiento cloud

En produccion, Agent A puede operar como microservicio sincrono para alto riesgo y como auditor asincrono para monitoreo masivo.
La arquitectura objetivo incluye API Gateway, vector store gobernado, motor de reglas, event bus, observabilidad y tablero de excepciones.
