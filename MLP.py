import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
import h5py
import numpy as np



#load data from hdf5 files
def load_data():
    train_dataset = h5py.File('trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test




# Neural network with multiple hidden layers 
def initialisation(dimensions):

    parametres={}
    C=len(dimensions)
    for i in range(1,C):
        parametres['w'+str(i)]=np.random.randn(dimensions[i],dimensions[i-1])
        parametres['b'+str(i)]=np.random.randn(dimensions[i],1)

    return parametres 



def segmoid(z):
    return 1/(1+np.exp(-z))



def forward_propagation(x,parametres):

    activations={'A0':x}
    C=len(parametres)//2

    for i in range(1,C+1):
        z=parametres['w'+str(i)].dot(activations['A'+str(i-1)]) +parametres['b'+str(i)]
        activations['A'+str(i)]=segmoid(z)


    return activations




def log_loss(a,y):
    m=len(y)
    ep=1e-15
    return 1/len(y) * np.sum(-y*np.log(a+ep) - (1-y)*np.log(1-a+ep))




def back_propagation(y,activations,parametres):
    m=y.shape[1]
    C=len(parametres)//2

    dZ=activations['A'+str(C)]-y
    gradients={}

    # we can use also for c in range(C,0,-1): this is equivalent to the previous line , give use c from C to 1
    for i in reversed(range(1,C+1)):
    
        gradients['dw'+str(i)]=1/m *np.dot(dZ,activations['A'+str(i-1)].T)
        gradients['db'+str(i)]=1/m *np.sum(dZ,axis=1,keepdims=True)
        if i>1:
            dZ=np.dot(parametres['w'+str(i)].T,dZ) * activations['A'+str(i-1)] * (1-activations['A'+str(i-1)])


    return gradients




def update(gradients,parametres,learning_rate):

    C=len(parametres)//2

    for i in range(1,C+1):
     parametres['w'+str(i)]=parametres['w'+str(i)]-learning_rate*gradients['dw'+str(i)]
     parametres['b'+str(i)]=parametres['b'+str(i)]-learning_rate*gradients['db'+str(i)]

    return parametres


def predict(x,parametres):
    k=len(parametres)//2
    activations=forward_propagation(x,parametres)
    A=activations['A'+str(k)]
    #y_pred=[1 if A2[i]>=0.5  else 0  for i in range(A2.shape[0])]
    #we can use this methode faster and easier
    return A>=0.5   #similar to return np.array(y_pred)



def plot_decision_boundary(x_train, y_train, parametres, title="Frontière de décision"):
    """
    Affiche la frontière de décision pour un modèle donné.
    
    Arguments :
    x_train    : np.array, shape (2, N) ou (N, 2)
    y_train    : np.array, labels (N,)
    predict    : fonction, prend en entrée X (shape (2, M)) et paramètres, retourne prédictions
    parametres : paramètres du modèle
    title      : titre du graphique
    """
    # S'assurer que x_train est transposé correctement
    
       # Création d'une grille pour visualiser la frontière
    x_t = x_train.T
    y_t = y_train.flatten()

    x_min, x_max = x_t[:, 0].min() - 0.5, x_t[:, 0].max() + 0.5
    y_min, y_max = x_t[:, 1].min() - 0.5, x_t[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),np.linspace(y_min, y_max, 500))

     #On empile la grille pour la passer dans ton modèle
    X_grid = np.c_[xx.ravel(), yy.ravel()].T   # -> (2, nombre_de_points)

     #Prédiction sur la grille avec ton propre réseau
    Z = predict(X_grid, parametres)            # ta fonction predict()
    Z = Z.reshape(xx.shape)                    # reshape pour affichage

     # Visualisation
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(x_t[:, 0], x_t[:, 1], c=y_t, s=30, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title("Frontière de décision de mon réseau de neurones (x_train.T, y_train.reshape(1,-1))")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()





def neural_network1(x,y,hidden_layers=(32,32,32),learning_rate=0.01,n_iter=100):
    
    np.random.seed(0)

    #initialisation of parametres W, b
    dimensions=[x.shape[0]]+list(hidden_layers)+[y.shape[0]]
    parametres=initialisation(dimensions)

    #history saving 
    train_loss=[]
    train_acc=[]

    for i in tqdm(range(n_iter)):

        activations=forward_propagation(x,parametres)
        gradients=back_propagation(y,activations,parametres)
        parametres=update(gradients,parametres,learning_rate)


        if i%10==0:
            C=len(parametres)//2

            train_loss.append(log_loss(activations['A'+str(C)],y))
            y_pred=predict(x,parametres)
            current_accuracy=accuracy_score(y.flatten(),y_pred.flatten())
            train_acc.append(current_accuracy)
        

    # result visualization
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title('Loss')
    plt.plot(train_loss,label='Train loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('Accuracy')
    plt.plot(train_acc,label='Train accuracy')
    plt.legend()
    plt.show()

        # plot decision boundary if input features are 2D
    plot_decision_boundary(x, y, parametres, title="Frontière de décision du réseau de neurones")  
























