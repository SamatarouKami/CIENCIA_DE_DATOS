import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from numpy import isnan, nan
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df2017 = pd.read_excel('2017.xlsx', index_col=0, header=0, sheet_name=[0,1,2,3,4])
df2017[0]['Año']='2017'
df2017[0]['Categoría']='Aficionado'
df2017[1]['Año']='2017'
df2017[1]['Categoría']='Juvenil'
df2017[2]['Año']='2017'
df2017[2]['Categoría']='Profesional'
df2017[3]['Año']='2017'
df2017[3]['Categoría']='SmarTIC'
df2017[4]['Año']='2017'
df2017[4]['Categoría']='Reto al guión'

df2017 = pd.concat([df2017[0],df2017[1],df2017[2],df2017[3],df2017[4]],sort=True)
#print(df2017.columns)

df2017 = df2017[['Año','Categoría','colombiano','genero','edad','sexo']]
df2017.set_axis(['Año','Categoría','País','Género','Edad','Sexo'], axis='columns', inplace=True)

#print(len(df2017))

clasificacion = df2017.Sexo.unique()
#print(clasificacion)


df2017['Sexo'] = df2017.Sexo.replace('Discapacidad visual',nan)
df2017['Sexo'] = df2017.Sexo.replace('Discapacidad auditiva',nan)


df2017 = df2017.dropna()



df2017.to_csv('clasificacion2017.csv', index=False)
#print(df2017)

#Preparando Datos

df = pd.read_csv("https://raw.githubusercontent.com/SamatarouKami/CIENCIA_DE_DATOS/master/clasificacion2017.csv")

#df = pd.read_csv("clasificacion2017.csv")
#cat = pd.Categorical(df.Categoría)
#df.Categoría = cat.codes

gen = pd.Categorical(df.Género)
df.Género = gen.codes

pai = pd.Categorical(df.País)
df.País = pai.codes

sex = pd.Categorical(df.Sexo) 
df.Sexo = pai.codes

#print(df.Sexo)



#df['etiquetas'] = [ 1 if df.Sexo[j] == 'Femenino' else 0 for j in df.Sexo.keys()] # Clasificar Sexo

df['etiquetas'] = [1 if df.Categoría[i] == 'Juvenil' else 0 for i in df.Categoría.keys()]# Clasificar Categorias
print(df.etiquetas.value_counts())


# importantes para Train_test_split

y = df.etiquetas

xVars = ['Edad', 'Sexo', 'Género'] 
x = df.loc[:, xVars].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components = 2) # pedimos uno bidimensional
X = pca.fit_transform(x)


#
'''
from math import ceil, sqrt
from numpy import isnan, nan, arange, meshgrid, c_
import matplotlib.pyplot as plt
h=0.2
# código de https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", \
         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes"]
classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), \
    SVC(gamma=2, C=1), GaussianProcessClassifier(1.0 * RBF(1.0)), \
    DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), \
    AdaBoostClassifier(), GaussianNB()]
k = int(ceil(sqrt(len(classifiers) + 1)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42) # división
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = meshgrid(arange(x_min, x_max, h), arange(y_min, y_max, 0.02))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.rcParams["figure.figsize"] = [16, 16]
figure = plt.figure()
ax = plt.subplot(k, k, 1)
ax.set_title("Datos de entrada")
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.2, edgecolors='k') # entrenamiento
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.2, edgecolors='k') # validación
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i = 2
for name, clf in zip(names, classifiers):
    ax = plt.subplot(k, k, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.3f' % score).lstrip('0'), size=40, horizontalalignment='right')
    i += 1
plt.tight_layout()
plt.show()


'''
# código de https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html 
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", \
         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes"]
classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025), \
    SVC(gamma=2, C=1), GaussianProcessClassifier(1.0 * RBF(1.0)), \
    DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), \
    AdaBoostClassifier(), GaussianNB()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42) # la misma división

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    print(name, clf.score(X_test, y_test))
    expected, predicted = y_test, clf.predict(X_test)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print('-' * 60)

