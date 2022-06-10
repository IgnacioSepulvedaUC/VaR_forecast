# Modelos de prediccion del VaR.

Este repositorio contiene los datos y modelos ocupados para predecir el Value at Risk.

## Modelos. 

**Historico**: Ocupa parametro historicos y por medio de una simulacion de trayectorias de un procesos browniano geometrico calcula el VaR asociado. 

**GARCH(1,1)**: Se asume una media constante, pero la varianza ahora varia en el tiempo. Por lo tanto se modela como una varianza condicional del tipo GARCH. 

**Conditional Autoregresive VaR**: Se modela directamente el quantil a elección mediante un proceso autoregresivo mas un componente que controla el nivel. 

**Gradient Boosting**: Se modifica la funcion de perdida tradicional para que modelo el quantil que deseamos predecir, se ocupan los rezagos de los retornos como input.

**Quantile Convolutional Neural Network**: Se intenta predecir ocupando CNN, se modifica la funcion de perdida para que entre un quantil. 





