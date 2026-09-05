import numpy as np
from keras.datasets import mnist

from keras.datasets import mnist



#data preporation

rng = np.random.default_rng()


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).T
X_test = X_test.reshape(-1, 784).T
X_train = X_train / 255.0
X_test = X_test / 255.0

def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((num_classes, labels.shape[0]))
    one_hot[labels, np.arange(labels.shape[0])] = 1
    return one_hot

Y_train = to_one_hot(y_train)
Y_test = to_one_hot(y_test)

# print("Данные готовы!")
# print(f"X_train shape: {X_train.shape}")  # (784, 60000)
# print(f"Y_train shape: {Y_train.shape}")  # (10, 60000)
# print(f"X_test shape:  {X_test.shape}")   # (784, 10000)
# print(f"Y_test shape:  {Y_test.shape}")   # (10, 10000)

# print(f"\nЗначения пикселей: min={X_train.min():.3f}, max={X_train.max():.3f}")
# print(f"Пример one-hot для первой картинки: {Y_train[:, 0]}")
# print(f"Правильная цифра: {np.argmax(Y_train[:, 0])}")

input_size = 784
hidden_size = 64 
output_size = 10 
batch_size = 32
#1 слой вь нем 64 нейрона

def relu(inp: np.ndarray ):
    return np.where(inp>0, inp, 0)

def softmax(z):
    # axis=0 означает вычисления по вертикали (вдоль столбцов)
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)#веса первого слоя
b1 = np.zeros(shape=(hidden_size, 1))#байас 1 слой

#z1 результат нейронов 1 слоя до релу (hidden_size * batch)
#a1 результат нейронов 1 слоя после релу (hidden_size * batch)

w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)#веса втоого слоя
b2 = np.zeros(shape=(output_size, 1)) #байас второй слой
#z2 результат нейронов 2 слоя до релу(outputsize, batch)
#a2 результат нейронов 2 слоя после релу(outputsize, batch)

def forward(x: np.ndarray):
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)


    z2 = np.dot(w2,a1) + b2
    ans = softmax(z2)
    return ans, z2, a1, z1


y_true = np.zeros(shape=(output_size, batch_size))# правильные ответы (one-hot) (output_size, batch_size)
def backwards(probs: np.ndarray, y_true:np.ndarray, a1: np.ndarray, z1:np.ndarray, x:np.ndarray ):
    m = y_true.shape[1]
    dz2 = (probs - y_true)/m
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    da1 = np.dot(w2.T, dz2)
    dz1 = da1.copy()
    dz1[z1<=0] = 0
    dw1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2


def update_weights(dw1, db1, dw2, db2, learning_rate):
    global w1, b1, w2, b2
    
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

def compute_loss(probs, y_true):
    epsilon = 1e-8  # чтобы не было log(0)
    log_probs = -np.log(probs + epsilon)
    loss = np.sum(y_true * log_probs) / y_true.shape[1]  # среднее по батчу
    return loss


def train(X_train, Y_train, epochs=30, learning_rate=0.1):
    """
    Обучает нейросеть на данных.
    
    Аргументы:
        X_train: (input_size, num_samples) — входные данные
        Y_train: (output_size, num_samples) — правильные ответы (one-hot)
        epochs: количество эпох
        learning_rate: шаг обучения
    
    Возвращает:
        history: список значений loss за каждую эпоху
    """
    
    # ==========================================
    # 1. ПОДГОТОВКА
    # ==========================================
    num_samples = X_train.shape[1]  
    batch_amount = (num_samples + batch_size - 1) // batch_size

    history = [] #loss history
    # ==========================================
    # 2. ЦИКЛ ПО ЭПОХАМ
    # ==========================================
    
    for epoch in range(epochs):
        #learning_rate = 0.1 * (0.9 ** (epoch // 5))
        ids = np.arange(0, num_samples)
        rng.shuffle(ids)

        X_shuffled = X_train[:, ids]
        Y_shuffled = Y_train[:, ids]

        total_loss = 0
        
        
        # 2.3. ЦИКЛ ПО БАТЧАМ
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]

            probs, z2,a1,z1 = forward(X_batch)
            total_loss += compute_loss(probs, Y_batch)
        
            dw1, db1, dw2, db2 = backwards(probs,Y_batch,a1,z1,X_batch)
            update_weights(dw1, db1, dw2, db2, learning_rate)    
            # Возьми батч из X_shuffled и Y_shuffled
            # Используй срез i : i+batch_size
            # Назови их X_batch и Y_batch+
            
            # Вызови forward и получи probs, z2, a1, z1+
            
            # Вычисли loss через compute_loss
            
            # Добавь loss к total_loss+
            
            # Вызови backwards и получи dw1, db1, dw2, db2
            
            # Вызови update_weights с learning_rate
            if epoch == 0 and i == 0:
                print("loss:", compute_loss(probs, Y_batch))
                print("dw1:", np.linalg.norm(dw1))
                print("dw2:", np.linalg.norm(dw2))
                print("w1:", np.linalg.norm(w1))
                print("w2:", np.linalg.norm(w2))
                    

        avg_loss = total_loss/batch_amount
        history.append(avg_loss)
        (print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"))
        # 2.4. КОНЕЦ ЭПОХИ
        # Вычисли средний loss (total_loss / num_batches)
        # Добавь его в history
        # Выведи информацию: эпоха, loss
        # (print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"))
    
    # ==========================================
    # 3. ВОЗВРАТ РЕЗУЛЬТАТА
    # ==========================================
    return history
    # Верни history (список loss по эпохам)

def evaluate(X_test, Y_test, num_samples=None):
    if num_samples is None:
        num_samples = X_test.shape[1]
    
    X_batch = X_test[:, :num_samples]
    Y_batch = Y_test[:, :num_samples]
    probs, _, _, _ = forward(X_batch)
    
    predictions = np.argmax(probs, axis=0)
    true_labels = np.argmax(Y_batch, axis=0)
    accuracy = np.mean(predictions == true_labels)
    
    return accuracy, predictions, true_labels


history = train(X_train, Y_train, epochs=30)
print(history)

accuracy, preds, labels = evaluate(X_test, Y_test, 1000)
print(f"Test accuracy: {accuracy*100:.2f}%")    