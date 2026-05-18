import time
import os
import psutil
import threading

class ColetorMetricas:
    """ 
    Coleta CPU e memória em background durante o benchmark, sem interferir com o
    tempo de execução da pipeline

    """

    def __init__(self, interval: float = 0.05):
        """
        Construtor da classe
        
        1. Define o intervalo de amostragem (50ms por padrão)
        2. Cria as listas de acumulação de valores
        3. Prende o coletor ao processo via 'psutil.Process(os.getpid()'

        """    
        self.interval = interval
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._stop = threading.Event()
        self._process = psutil.Process(os.getpid())

    def comecar(self):
        """
        Método que inicia o benchmark

        1. Limpa qualquer dado de uma rodada anterior e dispara a thread.

        """
        self._stop.clear()
        self._cpu_samples.clear()
        self._mem_samples.clear()
        self._thread = threading.Thread(target=self._coletar, daemon=True)
        self._thread.start()

    def parar(self):
        """
        Método de parar o benchmark

        1. Manda a thread parar com 'self._stop.set()', subindo uma flag thread-safe
        2. Aguarda ela terminar a coleta com 'self._thread.join()', garantindo a coleta total 
        
        """
        self._stop.set()
        self._thread.join()

    def _coletar(self):
        """
        Loop de coleta que roda em background

        1. A cada 50ms (por padrão) coleta:
            CPU em percentual do processo
            Memória em MB de quanto o processo ocupa da RAM   
        """
        while not self._stop.is_set():
            self._cpu_samples.append(self._process.cpu_percent(interval=None))
            self._mem_samples.append(self._process.memory_info().rss / (1024 ** 2))  # MB
            time.sleep(self.interval)

    @property
    def uso_medio_cpu(self) -> float:
        """
        Média de uso da CPU durante o processo

        1. Pega todas as samples e divide pela sua quantidade
        """
        return sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0

    @property
    def pico_memoria_mb(self) -> float:
        """
        Pico de uso de memória para saber o máximo que a biblioteca consumiu
        """

        return max(self._mem_samples) if self._mem_samples else 0.0
