import numpy as np
import matplotlib.pyplot as plt

def plot_svm(x, y, model):
        
    plt.figure(figsize=(10,10))
      
    # Dibujar datos de cada clase:
    plt.plot(x[y == 1, 0], x[y == 1, 1],'bo', markersize=9)
    plt.plot(x[y == -1, 0], x[y == -1, 1], 'ro',markersize=9)
        
    # Calcular limites
    xmin, xmax = x[:,0].min()-0.5, x[:,0].max()+0.5
    ymin, ymax = x[:,1].min()-0.5, x[:,1].max()+0.5  
    
    # Dibujar superficie de decisión y margenes
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), 
                         np.linspace(ymin, ymax, 100))
    data = np.array([xx.ravel(), yy.ravel()]).T
    zz = model.project(data).reshape(xx.shape)
    plt.contour(xx, yy, zz, [0.0], colors='k', linewidths=2)
    plt.contour(xx, yy, zz, [-1.0, 1.0], colors='grey',
                linestyles='--', linewidths=2)
    plt.contourf(xx, yy, zz, [min(zz.ravel()), 0.0, max(zz.ravel())],alpha=0.8, 
                 cmap=plt.cm.RdBu )
    
    # Dibujar vectores soporte
    plt.scatter(model.sv[:, 0], model.sv[:, 1], s=200, color="gold")  
    
    # Detallar figura
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20)
    plt.title('Support Vector Machine', fontsize=20)
    #plt.title(r'$Gamma =${0:3.3f}, $C=${1:3.3f}'.format(model.gamma, model.C), fontsize=15);   
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()