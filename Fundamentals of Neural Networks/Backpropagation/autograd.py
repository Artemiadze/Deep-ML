class Value:
    def __init__(self, data, _children=(), _op=''):
        """
        data — скалярное значение (float или int)
        grad — градиент (по умолчанию 0)
        _children — предыдущие узлы графа вычислений (для backprop)
        _op — операция, которая создала этот узел (для отладки)
        """
        self.data = data
        self.grad = 0
        self._backward = lambda: None  # функция обратного прохода
        self._prev = set(_children)    # зависимости
        self._op = _op                 # тип операции (например, '+', '*', 'ReLU')

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Реализация сложения двух Value (или Value и числа).
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Реализация умножения двух Value (или Value и числа).
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(out)/d(self) = other.data
            # d(out)/d(other) = self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """
        Реализация ReLU: f(x) = max(0, x)
        """
        out = Value(self.data if self.data > 0 else 0, (self,), 'ReLU')

        def _backward():
            # градиент проходит только если вход > 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Обратное распространение градиента по графу.
        """
        # Топологическая сортировка узлов графа
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Начальный градиент для выходного узла = 1
        self.grad = 1
        # Обратный проход
        for node in reversed(topo):
            node._backward()

a = Value(2)
b = Value(3)
c = Value(10)
d = a + b * c 
e = Value(7) * Value(2)
f = e + d
g = f.relu() 
g.backward() 
print(a,b,c,d,e,f,g)