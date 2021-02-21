# Perceptron tutorial
## Source:

youtube: perceptron in python - machine learning from scratch 06 - python tutorial
	 by Python Engineer

https://youtu.be/t2ym2a3pb_Y

## Summary:
	I have created a perceptron in python that does linear classfication(??) for two classes

## Linear model:
f(w,b) = w(transposed) * x + b

## Activation function:
### Unit step function:
g(x) = 1, if z >= theta; 0, otherwise

## Approximation function:
y_hat = g(f(w,b)) = g(w(transposed) * x + b)

## Perceptron training rule:
For each training sample x<sub>i</sub>
w := w + \Delta * w
\Delta w  = := alpha * (y<sub>i</sub> - y<sub>i</sub>hat) * x<sub>i</sub>

alpha: learning rate in [0,1]

## Notes:
* We make use of the inputs xn, input weights wn, and the activation function to get our output.
* The perceptron replicates an biological neouron. 
