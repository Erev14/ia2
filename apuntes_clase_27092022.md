Red nuronal multicapa MLP Dense N FC-NN
wi,j [l] peso sinaptico capa l de la neurona i, entrada
bi [l] sesgo capa l, neurona i
ai[l] Salida de la capa l, neurona i

w[l] ε R ^ (n_l * n_l-1)
b[l] ε R ^ l
a[l] ε R ^ n_l salida de la capa l
da ε R ^ n_l

tensor: 
0-tensor, 0 dim, escalar
1-tensor, 1 dim, vector
2-tensor, 2 dim, matriz


a[0] = x
for i = 1 to L
    z[i] = w[i] @ a[i-1] + b[i]
    a[i] = activate(z[i])
y_est = a[L]





Un problema de clasificación complejo, al ser mapeado de manera no lineal a un espacio de alta dimensión, es mas probable de ser linealmente separable que en baja dimensión, siempre que el espacio no sea densamente poblado" 
Tomas M. Cover, 1965
Teorema

Ruido de los datos determina la dimension a la que mapearla, muchas dimensiones aprendera tambien el ruido, pocas subentrena


Sea phi: R -> R una función no constante, acotada y continua. Sea F el conjunto de funciones continuas de variable real en el hipercubo [0,1]^m. Entonces para cualer epsilon > 0 función f E F tal que 
