import math
class Quaternion:
    def __init__(self,vector):
        if len(vector) == 4:
            self.r=vector[0]
            self.i=vector[1]
            self.j=vector[2]
            self.k=vector[3]
        else:
            raise ValueError("Quaternion must contain exactly 4 elements")
    def __repr__(self):
        return f"Quaternion({self.r}, {self.i}i, {self.j}j, {self.k}k)"
    def __add__(self, other):
        if isinstance(other,(int,float)):
            return Quaternion([self.r+other,self.i,self.j,self.k])
        elif isinstance(other, Quaternion):
            return Quaternion([self.r+other.r,self.i+other.i,self.j+other.j,self.k+other.k])
        else:
            return NotImplemented
    def __radd__(self, other):
        return self.__add__(other)#Сложение коммуитативно, поэтому можно не нагружать код
    def __sub__(self, other):
        if isinstance(other,(int,float)):
            return Quaternion([self.r-other,self.i,self.j,self.k])
        elif isinstance(other, Quaternion):
            return Quaternion([self.r-other.r,self.i-other.i,self.j-other.j,self.k-other.k])
        else:
            return NotImplemented
    def __rsub__(self, other):
        if isinstance(other,(int,float)):
            return Quaternion([other-self.r,-self.i,-self.j,-self.k])
        elif isinstance(other, Quaternion):
            return Quaternion([other.r-self.r,other.i-self.i,other.j-self.j,other.k-self.k])
        else:
            return NotImplemented
    def __mul__(self, other):
        if isinstance(other,(int,float)):
            return Quaternion([other * self.r, other * self.i, other * self.j, other * self.k])
        elif isinstance(other, Quaternion):
            a, b, c, d = self.r, self.i, self.j, self.k
            e, f, g, h = other.r, other.i, other.j, other.k
            return Quaternion([
                a * e - b * f - c * g - d * h,
                a * f + b * e + c * h - d * g,
                a * g - b * h + c * e + d * f,
                a * h + b * g - c * f + d * e
            ])
        else:
            return NotImplemented
    def __rmul__(self, other):
        if isinstance(other,(int,float)):
            return Quaternion([other * self.r, other * self.i, other * self.j, other * self.k])
        elif isinstance(other, Quaternion):
            a, b, c, d = other.r, other.i, other.j, other.k
            e, f, g, h = self.r, self.i, self.j, self.k
            return Quaternion([
                a * e - b * f - c * g - d * h,
                a * f + b * e + c * h - d * g,
                a * g - b * h + c * e + d * f,
                a * h + b * g - c * f + d * e
            ])
        else:
            return NotImplemented
    def conj(self):
        return Quaternion([self.r,-self.i,-self.j,-self.k])
    def norm(self):
        return (self.r**2+self.i**2+self.j**2+self.k**2)**0.5
    def normalize(self):
        n = self.norm()
        if n != 0:
            return Quaternion([self.r/n,self.i/n,self.j/n,self.k/n])
        else:
            raise ZeroDivisionError("Cannot normalize zero quaternion")
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other != 0:
                return Quaternion([self.r/other,self.i/other,self.j/other,self.k/other])
            else:
                raise ZeroDivisionError("Cannot divide by zero")
        elif isinstance(other, Quaternion):
            if abs(other.norm()) >= 1e-10:
                return (self*other.conj())/(other.norm()**2)
            else:
                raise ZeroDivisionError("Cannot divide by zero quaternion")
        else:
            return NotImplemented
    def __rtruediv__(self, other):
        if abs(self.norm()) >= 1e-10:
            if isinstance(other, (int, float,Quaternion)):
                return(other*self.conj())/(self.norm()**2)
            else:
                return NotImplemented
        else:
            raise ZeroDivisionError("Cannot divide by zero quaternion")

    def inverse(self):
        n2 = self.norm() ** 2
        if n2 == 0:
            raise ZeroDivisionError("Zero quaternion has no inverse")
        c = self.conj()
        return Quaternion([c.r / n2, c.i / n2, c.j / n2, c.k / n2])
    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return self.r == other.r and self.i == other.i and self.j == other.j and self.k == other.k
        if isinstance(other,(int,float)):
            return self.r == other and self.i == 0 and self.j == 0 and self.k == 0
        else:
            return False
    def __neg__(self):
        return Quaternion([-self.r,-self.i,-self.j,-self.k])
    def __abs__(self):
        return self.norm()
    def copy(self):
        return Quaternion([self.r, self.i, self.j, self.k])
    @property
    def vector(self):
        return (self.r, self.i, self.j, self.k)
    @property
    def vector3(self):
        return (self.i,self.j,self.k)
    def rotate_rad(self,axis,angle):
        if self.r != 0:
            raise ValueError("Can only rotate pure quaternions (real component must be 0)")
        if len(axis)!=3:
            raise ValueError("Axis must be 3 dimensional")
        L = Quaternion([0, *axis])
        L = L.normalize()*float(math.sin(angle/2))+float(math.cos(angle/2))
        return L*self*L.inverse()
    def rotate_deg(self, axis, angle):
        return self.rotate_rad(axis, math.radians(angle))



if __name__ == "__main__":
    q1 = Quaternion([1, 2, 3, 4])
    q2 = Quaternion([5, 6, 7, 8])

    print("q1 =", q1)
    print("q2 =", q2)

    print("\n=== Сумма ===")
    print("q1 + q2 =", q1 + q2)
    print("q1 + 10 =", q1 + 10)
    print("10 + q1 =", 10 + q1)

    print("\n=== Вычитание ===")
    print("q1 - q2 =", q1 - q2)
    print("q1 - 2 =", q1 - 2)
    print("2 - q1 =", 2 - q1)

    print("\n=== Умножение ===")
    print("q1 * q2 =", q1 * q2)
    print("q1 * 3 =", q1 * 3)
    print("3 * q1 =", 3 * q1)

    print("\n=== Деление ===")
    print("q1 / 2 =", q1 / 2)
    print("q1 / q2 =", q1 / q2)
    print("2 / q1 =", 2 / q1)

    print("\n=== Сопряженное ===")
    print("conj(q1) =", q1.conj())

    print("\n=== Норма ===")
    print("|q1| =", q1.norm())
    print("abs(q1) =", abs(q1))

    print("\n=== Нормализация ===")
    print("normalize(q1) =", q1.normalize())

    print("\n=== Обратное ===")
    print("inverse(q1) =", q1.inverse())

    print("\n=== Равенство ===")
    print("q1 == q1.copy() ->", q1 == q1.copy())
    print("q1 == q2 ->", q1 == q2)
    print("Quaternion([5,0,0,0]) == 5 ->",
          Quaternion([5, 0, 0, 0]) == 5)

    print("\n=== Умножение на -1 ===")
    print("-q1 =", -q1)

    print("\n=== Копирование ===")
    q3 = q1.copy()
    print("q3 =", q3)

    print("\n=== Вектор ===")
    print("q1.vector =", q1.vector)