class Network:
    def __init__(self, layers: list):
        self.layers = layers
        
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")
            elif e % 1000 == 0:
                print(f'epoch {e}, error {error}')