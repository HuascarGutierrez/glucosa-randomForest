import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#lectura de los datos
df = pd.read_csv('https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv')
#uso de las columnas glucose BMI y outcome
df = df.loc[:, ['Glucose', 'BMI', 'Outcome']]

# Filtrado de datos (BMI mayor a 50)
filtrada = df[df.BMI > 50]
#print(df.describe())

#GRAFICO 
# D: Diabetico (Outcome = 1)
# C: Control (Outcome = 0)

D = df[df.Outcome == 1]
C = df[df.Outcome == 0]

#Creacion del grafico
''' Descomentar para ver el grafico
plt.figure(figsize=(8, 6))

plt.scatter(C.Glucose, C.BMI, color='blue', label='Control')
plt.scatter(D.Glucose, D.BMI, color='red', label='Diabetic')

plt.xlabel('Glucose')
plt.ylabel('BMI')

plt.legend()
plt.show()
'''

#Separar en x e y
# x: matriz de caracteristicas
# y: vector de etiquetas (salida)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
#print(y)

#estandarizado de datos
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#print(x_scaled)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#entremaniento del modelo
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()
randomForest.fit(x_train, y_train)

#visualizacion de uno de los arboles
from sklearn import tree
'''Descomentar para ver el arbol
plt.figure(figsize=(10, 10))    
tree.plot_tree(randomForest.estimators_[0], filled=True)
plt.show()'''

#predicciones de los datos de prueba
y_pred_rf = randomForest.predict(x_test)
#print(y_pred_rf)

y_pred_proba_rf = randomForest.predict_proba(x_test)[:,1]
#print(y_pred_proba_rf)

#print(x_test[0])
#print(randomForest.predict_proba([x_test[0]]))

#guardar el modelo
joblib.dump(randomForest, 'modelo.pkl')
joblib.dump(scaler, "scaler.pkl")  # Guarda el scaler también

'''#probar el modelo
x_prediccion = x[0:5]
#print('Resultado de la prediccion')
prueba_scaled = scaler.fit_transform(x_prediccion)
print(x_prediccion)
print(prueba_scaled)
print(randomForest.predict_proba(prueba_scaled))
'''