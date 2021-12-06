import numpy as np
class KNearestNeighbor():
    def __init__(self, k):
        self.k = k

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[: self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))
        return y_pred

    def predict(self, x_test, num_loops=1):
        if num_loops == 2:
            distances = self.compute_distance_two_loops(x_test)
        if num_loops == 1:
            distances = self.compute_distance_one_loop(x_test)
        return self.predict_labels(distances)

    def compute_distance_vectorized(self, x_test):
        x_test_squared = np.sum(x_test ** 2, axis=1, keepdims=True)
        x_train_squared = np.sum(self.x_train ** 2, axis=1, keepdims=True)
        two_x_test_x_train = np.dot(x_test, self.x_train.T)
        return np.sqrt(x_train_squared - 2 * two_x_test_x_train + x_train_squared.T)

    def compute_distance_one_loop(self, x_test):
        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            distances[i, :] = np.sqrt(np.sum((self.x_train - x_test[i, :]) ** 2, axis=1))
        return distances

    def compute_distance_two_loops(self, x_test):
        # naive inefficient way
        num_test = x_test.shape[0]
        num_train = self.x_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distances[i, j] = np.sqrt(np.sum((np.sum(x_test[i, :] - self.x_train[j, :])) ** 2))
        return distances




if __name__ == '__main__':
    train = np.random.randn(1, 4)
    test = np.random.randn(1, 4)
    num_examples = train.shape[0]

    distance = np.sqrt(np.sum(test**2, axis=1, keepdims=True) + np.sum(train**2, keepdims=True, axis=1) - 2 * np.sum(test * train))

    # X = np.loadtxt('example_data/data.txt', delimiter=',')
    # y = np.loadtxt('example_data/targets.txt')
    KNN = KNearestNeighbor(k=3)
    KNN.train(train, np.zeros((num_examples)))
    corr_distance = KNN.compute_distance_two_loops(test)
    # print(f'accuracy: {sum(y_pred==y)/y.shape[0]}')
    print(f'the difference is {np.sum(np.sum((corr_distance - distance) ** 2))}')