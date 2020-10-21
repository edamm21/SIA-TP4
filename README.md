# TP4 SIA

## Instrucciones de configuración y ejecución
Para usar las librerías requeridas, utilizar los siguientes comandos:
```javascript
pip3 install numpy
pip3 install matplotlib
```

## Ejercicio 1
Para configurar el ejercicio 1, debemos especificar en el archivo ```input.json``` cuál método utilizar y los parámetros a utilizar. METHOD puede ser Kohonen u Oja, LEARNING_RATE especifica la tasa de aprendizaje inicial, R_NEIGHBORHOOD indica el tamaño de la matriz de Kohonen, y DATA_PATH especifica la ubicación del csv a utilizar. Ejemplo:

```javascript
    {
	    "METHOD":"kohonen",
	    "LEARNING_RATE":0.5,
	    "R_NEIGHBORHOOD": 5,
	    "DATA_PATH":"src/Ej1/data/europe.csv"
    }
```

Luego para ejecutar el programa, dentro de la carpeta ```src```, corremos el siguiente comando:

```javascript
python3 main.py
```

## Ejercicio 2
Para ejecutar el ejercicio 2, basta con abrir el archivo ```index.html``` con un navegador, o acceder a ```http://hopfield.herokuapp.com``` y seguir las instrucciones que aparrecen en pantalla.