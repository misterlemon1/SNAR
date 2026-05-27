import math
import unittest

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
        return self.__add__(other)  #Сложение коммутативно, поэтому можно не нагружать код
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
        """Возвращает сопряжённый кватернион (меняет знак мнимых частей)."""
        return Quaternion([self.r,-self.i,-self.j,-self.k])
    def norm(self):
        """Возвращает норму кватерниона."""
        return (self.r**2+self.i**2+self.j**2+self.k**2)**0.5
    def normalize(self):
        """Возвращает кватернион единичной нормы. Бросает ZeroDivisionError для нулевого кватерниона."""
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
        """Возвращает обратный кватернион q^-1 кватерниону q. (q*^-1=q^-1*q=1)"""
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
        """Возвращает копию кватерниона."""
        return Quaternion([self.r, self.i, self.j, self.k])
    @property
    def vector(self):
        """Возвращает кватернион в виде вектора с действительной частью. (4 компоненты)"""
        return (self.r, self.i, self.j, self.k)
    @property
    def vector3(self):
        """Возвращает мнимую часть кватерниона. (3 компоненты)"""
        return (self.i,self.j,self.k)
    def rotate_rad(self,axis,angle):
        """Вращение кватерниона вдоль оси на заданный угол в радианах"""
        if self.r != 0:
            raise ValueError("Can only rotate pure quaternions (real component must be 0)")
        if len(axis)!=3:
            raise ValueError("Axis must be 3 dimensional")
        L = Quaternion([0, *axis])
        L = L.normalize()*float(math.sin(angle/2))+float(math.cos(angle/2))
        return L*self*L.inverse()
    def rotate_deg(self, axis, angle):
        """Вращение кватерниона вдоль оси на заданный угол в градусах"""
        return self.rotate_rad(axis, math.radians(angle))

class IMU:
    """
    Имитация инерциального измерительного модуля. (Internal Measurement Unit)
    Интегрирует угловую скорость (рад/с) в кватернион ориентации методом Эйлера.
    Начальная ориентация — единичный кватернион с нулевой мнимой частью.
    """
    def __init__(self):
        self.direction = Quaternion([1,0,0,0])  #Начальное положение. Cos(0)=1, Sin(0)=0
    def update(self,wx,wy,wz,dt):
        """
        wx, wy, wz — угловая скорость по соответствующим осям (рад/с).
        dt — временной шаг между измерениями (с).
        Возвращает обновленный вектор направления.
        """
        omega = Quaternion([0,wx*dt/2,wy*dt/2,wz*dt/2])
        self.direction = self.direction+self.direction*omega  # dq/dt = 0.5 * q * omega_q  интегрирование методом Эйлера
        return self.direction
    def to_euler(self):
        """Углы Эйлера в радианах. Конвенция ZYX: объект поворачивается сначала вокруг Z, затем Y, затем X."""
        r,i,j,k = self.direction.vector
        qx = math.atan2(2*(r*i + j*k), 1 - 2*(i*i + j*j))
        qy = math.asin(max(-1, min(1, 2*(r*j - k*i))))
        qz = math.atan2(2*(r*k + i*j), 1 - 2*(j*j + k*k))
        return (qx,qy,qz)
    def to_euler_deg(self):
        """То же что to_euler(), но в градусах."""
        return tuple(math.degrees(a) for a in self.to_euler())


class TestQuaternion(unittest.TestCase):
    def setUp(self):
        self.q1 = Quaternion([1, 2, 3, 4])
        self.q2 = Quaternion([5, 6, 7, 8])
        self.identity = Quaternion([1, 0, 0, 0])
    # Создание
    def test_init_valid(self):
        """Корректное создание кватерниона."""
        q = Quaternion([1, 2, 3, 4])
        self.assertEqual(q.r, 1)
        self.assertEqual(q.i, 2)
        self.assertEqual(q.j, 3)
        self.assertEqual(q.k, 4)
    def test_init_invalid(self):
        """Неверное количество элементов бросает ValueError."""
        self.assertRaises(ValueError, Quaternion, [1, 2, 3])
        self.assertRaises(ValueError, Quaternion, [1, 2, 3, 4, 5])
        self.assertRaises(ValueError, Quaternion, [])
    # Сложение
    def test_add_quaternion(self):
        self.assertEqual(self.q1 + self.q2, Quaternion([6, 8, 10, 12]))
    def test_add_scalar(self):
        self.assertEqual(self.q1 + 10, Quaternion([11, 2, 3, 4]))
    def test_radd_scalar(self):
        """Сложение коммутативно — 10 + q должно работать."""
        self.assertEqual(10 + self.q1, Quaternion([11, 2, 3, 4]))
    def test_add_commutativity(self):
        """q1 + q2 == q2 + q1."""
        self.assertEqual(self.q1 + self.q2, self.q2 + self.q1)
    # Вычитание
    def test_sub_quaternion(self):
        self.assertEqual(self.q1 - self.q2, Quaternion([-4, -4, -4, -4]))
    def test_sub_scalar(self):
        self.assertEqual(self.q1 - 1, Quaternion([0, 2, 3, 4]))
    def test_rsub_scalar(self):
        """2 - q: вещественная часть вычитается, мнимые меняют знак."""
        self.assertEqual(2 - self.q1, Quaternion([1, -2, -3, -4]))
    def test_sub_not_commutative(self):
        """Вычитание некоммутативно."""
        self.assertNotEqual(self.q1 - self.q2, self.q2 - self.q1)
    # Умножение
    def test_mul_scalar(self):
        self.assertEqual(self.q1 * 3, Quaternion([3, 6, 9, 12]))
    def test_rmul_scalar(self):
        self.assertEqual(3 * self.q1, Quaternion([3, 6, 9, 12]))
    def test_mul_quaternion(self):
        """Проверка по формуле умножения кватернионов."""
        result = self.q1 * self.q2
        self.assertEqual(result, Quaternion([-60, 12, 30, 24]))
    def test_mul_not_commutative(self):
        """Умножение кватернионов некоммутативно."""
        self.assertNotEqual(self.q1 * self.q2, self.q2 * self.q1)
    def test_mul_identity(self):
        """Умножение на единичный кватернион не меняет объект."""
        self.assertEqual(self.q1 * self.identity, self.q1)
        self.assertEqual(self.identity * self.q1, self.q1)
    def test_mul_associativity(self):
        """Умножение ассоциативно: (q1*q2)*q3 == q1*(q2*q3)."""
        q3 = Quaternion([1, -1, 2, -2])
        left = (self.q1 * self.q2) * q3
        right = self.q1 * (self.q2 * q3)
        self.assertAlmostEqual(left.r, right.r, places=9)
        self.assertAlmostEqual(left.i, right.i, places=9)
        self.assertAlmostEqual(left.j, right.j, places=9)
        self.assertAlmostEqual(left.k, right.k, places=9)
    # Деление
    def test_div_scalar(self):
        result = Quaternion([2, 4, 6, 8]) / 2
        self.assertEqual(result, Quaternion([1, 2, 3, 4]))
    def test_div_by_zero_scalar(self):
        self.assertRaises(ZeroDivisionError, lambda: self.q1 / 0)
    def test_div_quaternion(self):
        """q / q должно давать единичный кватернион."""
        result = self.q1 / self.q1
        self.assertAlmostEqual(result.r, 1.0, places=9)
        self.assertAlmostEqual(result.i, 0.0, places=9)
        self.assertAlmostEqual(result.j, 0.0, places=9)
        self.assertAlmostEqual(result.k, 0.0, places=9)
    def test_div_by_zero_quaternion(self):
        zero = Quaternion([0, 0, 0, 0])
        self.assertRaises(ZeroDivisionError, lambda: self.q1 / zero)
    def test_rdiv_scalar(self):
        """2 / q — проверяем что работает."""
        result = 1 / self.identity
        self.assertAlmostEqual(result.r, 1.0, places=9)
        self.assertAlmostEqual(result.i, 0.0, places=9)
    # Сопряжение, норма и нормализация
    def test_conj(self):
        self.assertEqual(self.q1.conj(), Quaternion([1, -2, -3, -4]))
    def test_conj_double(self):
        """Двойное сопряжение возвращает исходный кватернион."""
        self.assertEqual(self.q1.conj().conj(), self.q1)
    def test_norm(self):
        self.assertAlmostEqual(self.q1.norm(), math.sqrt(1+4+9+16), places=9)
    def test_abs_equals_norm(self):
        """abs(q) должен совпадать с q.norm()."""
        self.assertEqual(abs(self.q1), self.q1.norm())
    def test_normalize_unit_norm(self):
        """Нормализованный кватернион имеет норму 1."""
        self.assertAlmostEqual(self.q1.normalize().norm(), 1.0, places=9)
    def test_normalize_zero_raises(self):
        zero = Quaternion([0, 0, 0, 0])
        self.assertRaises(ZeroDivisionError, zero.normalize)
    def test_norm_of_product(self):
        """Норма произведения равна произведению норм: |q1*q2| == |q1|*|q2|."""
        self.assertAlmostEqual(
            (self.q1 * self.q2).norm(),
            self.q1.norm() * self.q2.norm(),
            places=9
        )
    # Обратный кватернион
    def test_inverse(self):
        """q * q^-1 должно давать единичный кватернион."""
        inv = self.q1.inverse()
        result = self.q1 * inv
        self.assertAlmostEqual(result.r, 1.0, places=9)
        self.assertAlmostEqual(result.i, 0.0, places=9)
        self.assertAlmostEqual(result.j, 0.0, places=9)
        self.assertAlmostEqual(result.k, 0.0, places=9)
    def test_inverse_zero_raises(self):
        zero = Quaternion([0, 0, 0, 0])
        self.assertRaises(ZeroDivisionError, zero.inverse)
    def test_inverse_of_unit(self):
        """Обратный к единичному — он сам."""
        inv = self.identity.inverse()
        self.assertAlmostEqual(inv.r, 1.0, places=9)
        self.assertAlmostEqual(inv.i, 0.0, places=9)
    #Равенство и отрицание
    def test_eq_quaternion(self):
        self.assertEqual(self.q1, Quaternion([1, 2, 3, 4]))
        self.assertNotEqual(self.q1, self.q2)
    def test_eq_scalar(self):
        """Кватернион равен числу если мнимая часть нулевая."""
        self.assertEqual(Quaternion([5, 0, 0, 0]), 5)
        self.assertNotEqual(self.q1, 1)
    def test_neg(self):
        self.assertEqual(-self.q1, Quaternion([-1, -2, -3, -4]))
    def test_double_neg(self):
        """Двойное отрицание возвращает исходный кватернион."""
        self.assertEqual(-(-self.q1), self.q1)
    #Копирование и свойства
    def test_copy_equal_but_independent(self):
        """Копия равна оригиналу, но является отдельным объектом."""
        q_copy = self.q1.copy()
        self.assertEqual(q_copy, self.q1)
        self.assertIsNot(q_copy, self.q1)
    def test_vector_property(self):
        self.assertEqual(self.q1.vector, (1, 2, 3, 4))
    def test_vector3_property(self):
        self.assertEqual(self.q1.vector3, (2, 3, 4))
    #Вращение
    def test_rotate_identity(self):
        """Поворот на 0 радиан не меняет вектор."""
        v = Quaternion([0, 1, 0, 0])
        result = v.rotate_rad([0, 1, 0], 0)
        self.assertAlmostEqual(result.i, 1.0, places=9)
        self.assertAlmostEqual(result.j, 0.0, places=9)
        self.assertAlmostEqual(result.k, 0.0, places=9)
    def test_rotate_90_degrees(self):
        """Поворот вектора (1,0,0) вокруг Z на 90° даёт (0,1,0)."""
        v = Quaternion([0, 1, 0, 0])
        result = v.rotate_rad([0, 0, 1], math.pi / 2)
        self.assertAlmostEqual(result.i, 0.0, places=9)
        self.assertAlmostEqual(result.j, 1.0, places=9)
        self.assertAlmostEqual(result.k, 0.0, places=9)
    def test_rotate_rad_equals_rotate_deg(self):
        """rotate_rad и rotate_deg дают одинаковый результат."""
        v = Quaternion([0, 1, 2, 3])
        axis = [1, 0, 0]
        r1 = v.rotate_rad(axis, math.pi / 3)
        r2 = v.rotate_deg(axis, 60)
        self.assertAlmostEqual(r1.i, r2.i, places=9)
        self.assertAlmostEqual(r1.j, r2.j, places=9)
        self.assertAlmostEqual(r1.k, r2.k, places=9)
    def test_rotate_nonzero_real_raises(self):
        """Нельзя вращать кватернион с ненулевой вещественной частью."""
        self.assertRaises(ValueError, self.q1.rotate_rad, [0, 1, 0], 1.0)
    def test_rotate_wrong_axis_raises(self):
        """Ось вращения должна быть трёхмерной."""
        v = Quaternion([0, 1, 0, 0])
        self.assertRaises(ValueError, v.rotate_rad, [1, 0], 1.0)
    def test_rotate_preserves_norm(self):
        """Вращение не меняет норму вектора."""
        v = Quaternion([0, 1, 2, 3])
        result = v.rotate_rad([1, 1, 0], math.pi / 4)
        self.assertAlmostEqual(abs(result), abs(v), places=9)


class TestIMU(unittest.TestCase):
    def setUp(self):
        """Создаём свежий IMU перед каждым тестом."""
        self.imu = IMU()
        self.dt = 0.01
    def test_initial_state(self):
        """Начальная ориентация — единичный кватернион, все углы нулевые."""
        self.assertEqual(self.imu.direction, Quaternion([1, 0, 0, 0]))
        roll, pitch, yaw = self.imu.to_euler()
        self.assertAlmostEqual(roll,  0, places=9)
        self.assertAlmostEqual(pitch, 0, places=9)
        self.assertAlmostEqual(yaw,   0, places=9)
    def test_norm_stays_close_to_one(self):
        """Норма кватерниона не должна сильно уплывать от 1 за 1000 шагов."""
        for _ in range(1000):
            self.imu.update(0.1, 0.2, 0.3, self.dt)
        self.assertAlmostEqual(self.imu.direction.norm(), 1.0, places=2)
    def test_rotation_around_z(self):
        """Поворот только вокруг Z меняет yaw, roll и pitch остаются нулевыми."""
        omega_z = math.pi / 2  # 90°/с, за 100 шагов по 0.01с = 90°
        for _ in range(100):
            self.imu.update(0, 0, omega_z, self.dt)
        roll, pitch, yaw = self.imu.to_euler()
        self.assertAlmostEqual(math.degrees(roll),  0.0,  delta=0.1)
        self.assertAlmostEqual(math.degrees(pitch), 0.0,  delta=0.1)
        self.assertAlmostEqual(math.degrees(yaw),  90.0,  delta=0.5)
    def test_rotation_around_x(self):
        """Поворот только вокруг X меняет roll, pitch и yaw остаются нулевыми."""
        omega_x = math.pi / 4  # 45°/с, за 100 шагов = 45°
        for _ in range(100):
            self.imu.update(omega_x, 0, 0, self.dt)
        roll, pitch, yaw = self.imu.to_euler()
        self.assertAlmostEqual(math.degrees(roll),  45.0, delta=0.5)
        self.assertAlmostEqual(math.degrees(pitch),  0.0, delta=0.1)
        self.assertAlmostEqual(math.degrees(yaw),    0.0, delta=0.1)
    def test_euler_and_euler_deg_consistent(self):
        """to_euler() и to_euler_deg() должны давать одинаковый результат."""
        self.imu.update(0.3, 0.5, 0.1, 0.5)
        rad = self.imu.to_euler()
        deg = self.imu.to_euler_deg()
        for r, d in zip(rad, deg):
            self.assertAlmostEqual(math.degrees(r), d, places=9)
    def test_full_rotation_returns_to_start(self):
        """Полный оборот вокруг Y возвращает к начальной ориентации."""
        omega_y = 2 * math.pi  # 360°/с, за 1000 шагов по 0.001с = 360°
        for _ in range(1000):
            self.imu.update(0, omega_y, 0, 0.001)
        roll, pitch, yaw = self.imu.to_euler()
        self.assertAlmostEqual(math.degrees(roll),  0.0, delta=0.05)
        self.assertAlmostEqual(math.degrees(pitch), 0.0, delta=0.05)
    def test_update_returns_quaternion(self):
        """update() должен возвращать кватернион."""
        result = self.imu.update(0.1, 0.2, 0.3, self.dt)
        self.assertIsInstance(result, Quaternion)



if __name__ == "__main__":
    unittest.main(verbosity=2)  # verbosity=2 — подробный вывод