Maquina de Soporte Vectorial (SVM)

Vapnik (Aprendizaje Estadistico)

w^T X + b > 0 -> X+
w^T X + b < 0 -> X-


L= Ax+By+c
P= (x, y)

dist(L, P) = Ax+By+c / (sqr(A^2+B^2))

w^T X+ + b >= 1 y = 1
w^T X- + b < -1  y = -1

y_i(w^T x_i + b) >= 1

y_i(w^T x_i + b) - 1 >= 0

VS (asegura que todos los puntos cumplen la igualdad)
y_i(w^T x_i + b) - 1 = 0


d = x+ dot(w/||w||) - x- dot(w/||w||) = (x+ - x-) dot(w/||w||)

Para X+
(1)(w^Tx++b) - 1 = 0
w^Tx+ =  1 - b
x+ dot(w) =  1 - b

Para X-
(-1)(w^Tx_++b) - 1 = 0
-w^Tx_-b =  1
-w^Tx_ =  1 + b
x- dot(w) =  - 1 - b

d = x+ dot(w/||w||) - x- dot(w/||w||)
= (x+ - x-) dot(w/||w||)
= (x+ dot(w) - x- dot(w)) 1/||w||
= ((1-b) - (-1-b)) 1/||w||
= 2 / ||w||


buscando maximizar el margen

