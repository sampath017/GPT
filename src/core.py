class Scalar:
    def __init__(self, data, _children=(), _operation=""):
        self.data = data

        # Private fields
        self._children = _children
        self._operation = _operation

    def __add__(self, other):
        other = Scalar._check(other)
        out = Scalar(self.data + other.data,
                     _children=(self, other), _operation="+")

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = Scalar._check(other)
        out = Scalar(self.data + other.data,
                     _children=(self, other), _operation="*")

        return out

    def __rmul__(self, other):
        return self + other

    # Private methods
    @staticmethod
    def _check(other):
        return Scalar(other) if not isinstance(other, Scalar) else other

    def __repr__(self):
        return f"Scalar(data={self.data})"
