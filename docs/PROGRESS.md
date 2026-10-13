# Progreso

## Estado actual

Estudio terminado en local: 16 entrenamientos (LightGBM ×7, char-CNN ×3, BiGRU ×3, DistilBERT ×3),
evaluación con bootstrap por dominio, ablaciones, figuras, README y model card regenerados desde
`reports/results.json`; tests en verde. Siguiente paso: auditoría de publicación (sanitizer) y
push a `Yoyagm/phishing-url-small-models`, que hace la integración.
Bloqueos: ninguno.

## Registro

### 2026-10-13

- **Hecho:** repo creado; código de datos de phishing-url-detector vendorizado byte a byte (hashes
  fijados en `tests/test_vendored.py`). El LightGBM reproducido coincide con el publicado al último
  dígito (ROC-AUC 0,9372024471270224 y recall 0,705472034654096 en OOD; 0,5852842809364549 en
  OpenPhish por filas), y `make eval` se niega a seguir si deja de coincidir.
- **Resultado:** la respuesta a la pregunta es **no**. Fuera de distribución, a FPR 1 %, DistilBERT
  empata con LightGBM (71,1 % vs 70,5 %, diferencia pareada +0,6 [−6,7; +6,1]); char-CNN pierde
  claramente y BiGRU pierde contra el control. Dentro de distribución los tres neuronales ganan por
  4-6 puntos con IC lejos de cero. La pérdida se concentra en el extremo estricto de la curva ROC
  (a FPR 0,1 % LightGBM 50,8 %, DistilBERT 44,5 %, modelos de caracteres ~23 %).
- **Decisión:** bootstrap por dominio registrable (no por host) y emparejado entre modelos; se
  descartó el bootstrap por host porque el hold-out interno concentra el 38,5 % de su phishing en
  dos dominios (`web.app`, `repl.co`) y un IC por host sería falsamente estrecho.
- **Decisión:** dos referencias LightGBM además de la publicada (mismas filas con semillas 0-2, y
  las filas exactas de los modelos neuronales). Se añadieron al ver que la semilla sola mueve el
  recall OOD ~2 puntos y que el 70,5 % publicado es la peor de sus cuatro semillas.
- **Decisión:** DistilBERT con ajuste completo (no LoRA): 67 M parámetros caben de sobra en 16 GB;
  máximo 3 épocas (todas las semillas pararon en la 2.ª, mejor época la 1.ª; ~19 min por semilla).
  También se midió ELECTRA-small (2-3× más rápido); se eligió DistilBERT porque cabía en el
  presupuesto de tiempo y es el modelo que nombra la spec.
- **Hallazgo técnico:** PyTorch y LightGBM no pueden convivir en un proceso en macOS (segfault o
  bloqueo según el orden de import; dos runtimes OpenMP). `train.py` los importa en perezoso y el
  test del baseline corre en un subproceso.
- **Hallazgo técnico:** en MPS, agrupar los lotes por longitud (pools de 50 lotes) bajó el paso de
  DistilBERT de 989 a 320 ms y el de ELECTRA-small de 354 a 150 ms.
