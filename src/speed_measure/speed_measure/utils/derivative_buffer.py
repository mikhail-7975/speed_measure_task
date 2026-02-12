import numpy as np

def is_standard_dtype(dtype):
    dtype = np.dtype(dtype)
    return (
        dtype.names is None          # Not a structured dtype
        and dtype.subdtype is None   # Not an array-in-dtype (e.g., '(3,)f4')
        and dtype.kind != 'O'        # Not object dtype
        and dtype.kind != 'U'        # Not Unicode string (optional filter)
        and dtype.kind != 'S'        # Not byte string (optional filter)
    )

class DerivativeBuffer:
    """
    Буфер из 3 значений float32 с временными метками для точного вычисления производной.
    Хранит пары (timestamp, value), вычисляет производную как Δvalue/Δtime.
    """
    
    def __init__(self, data_dtype):
        if not is_standard_dtype(data_dtype):
            raise Exception("DerivativeBuffer exception: use numpy dtype like np.float32, np.float64 and so on")
        self.values = np.zeros(3, dtype=data_dtype)      # [t-2, t-1, t]
        self.timestamps = np.zeros(3, dtype=np.float64)  # соответствующие временные метки
        self.count = 0  # количество добавленных измерений
    
    def append(self, value: np.float64, timestamp: np.float64) -> None:
        """
        value: измеренное значение
        timestamp: временная метка (в секундах, мс и т.д.)
        """
        self.values = np.roll(self.values, -1)
        self.timestamps = np.roll(self.timestamps, -1)
        
        self.values[-1] = value
        self.timestamps[-1] = timestamp
        
        self.count = min(self.count + 1, 3)
    
    def derivate(self) -> float:
        """
        Вычислить производную на основе реальных временных интервалов.
        
        Стратегия:
        - < 2 точек: возвращает 0.0 (недостаточно данных)
        - 2 точки: прямая разность (v₂ - v₁) / (t₂ - t₁)
        - 3 точки: центральная разность (v₂ - v₀) / (t₂ - t₀)
        
        Returns:
            float: оценка производной (значение/время)
        """
        if self.count < 2:
            return 0.0
        
        if self.count == 2:
            dt = self.timestamps[2] - self.timestamps[1]
            if np.isclose(dt, 0.0):
                return 0.0
            return float((self.values[2] - self.values[1]) / dt)
        
        dt = self.timestamps[2] - self.timestamps[0]
        if np.isclose(dt, 0.0):
            return 0.0
        return float((self.values[2] - self.values[0]) / dt)
    
    def get_value(self) -> float:
        return float(self.values[-1]) if self.count > 0 else 0.0
    

