import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from Individual import Individual
import random

class Evolution:
    """
    Класс, описывающий эволюцию популяции.
    n - количество устройств
    task_vector - вектор задач
    p_mut - вероятность мутации
    p_cross - вероятность кроссовера
    population_size - размер популяции
    k - количество поколений
    model - модель (1 - Гольдберг, 2 - Холланд)
    """
    def __init__(self, n: int, task_vector: np.ndarray, p_mut: float, p_cross: float, population_size: int = 5, k: int = 3, model: int = 1) -> None:
        self.population_size = population_size
        task_matrix = self.get_x_gens(np.array(task_vector), n)
        self.population = [Individual(n, task_vector, task_matrix, p_mut, p_cross, i+1, model) for i in range(self.population_size)]
        self.num_generations = k
        self.generation_buffer = {} if model == 1 else []
        self.model = model
    
    # Нужно для симуляции устройств, которые не могут считать определенные задачи.
    # Тут вектор задач превращается в матрицу, и добавляет случайно значения -1.
    # -1 означает, что устройство не может считать эту задачу
    def get_x_gens(self, task_matrix: np.ndarray, n: int) -> np.ndarray:
        task_matrix_t = task_matrix.reshape(-1, 1)
        repeated_matrix = np.tile(task_matrix_t, (1, n))

        rows, cols = repeated_matrix.shape
        for i in range(rows):
            indices = np.arange(cols)
            np.random.shuffle(indices)
            num_neg_ones = np.random.randint(1, cols)
            
            for j in range(num_neg_ones):
                if repeated_matrix[i, indices[j]] != -1:
                    repeated_matrix[i, indices[j]] = -1

        return repeated_matrix
        
    def get_random_individual(self, current_ind: Individual) -> Individual:
        while True:
            random_ind = random.choice(self.population)
            if random_ind != current_ind:
                return random_ind

    def print_target_func(self, inds: List[Individual]) -> None:
        inds_num = [ind.num for ind in inds]
        targets = [sum(ind.phenotype['max_feno']) for ind in inds]
        print("\t" + "Номер особи:" + " ".join(map(str, inds_num)))
        print("\t" + "Целевая функция:" + " ".join(map(str, targets)))

    def print_population(self) -> None:
        for ind in self.population:
            print(f"Особь {self.population.index(ind) + 1}")
            ind.print_info()
           
    # Основной метод, описывающий создания особей в популяции 
    def get_childs(self, ind1: Individual, ind2: Individual) -> List[Individual]:
        childs = ind1.crossover(ind2)
        if childs is not None:
            for ind in childs:
                ind.mutation()
                ind.update_pheno()
                ind.isChild = True
                
            ind1.update_pheno()
            childs.append(ind1)
            return childs
        else:
            print("Идёт мутация:")
            mut_ind = Individual(ind1.n, ind1.x_gens, ind1.full_x_gens, ind1.p_mut, ind1.p_cross, num=ind1.num, model=ind1.model)
            mut_ind.y_gens = ind1.y_gens.copy()
            mut_ind.mutation()
            mut_ind.update_pheno()
            
            if sum(mut_ind.phenotype["max_feno"]) < sum(ind1.phenotype["max_feno"]):
                print("Фенотип изменился в лучшую сторону")
                print("Было:", ind1.phenotype["max_feno"], sum(ind1.phenotype["max_feno"]))
                print("Стало:", mut_ind.phenotype["max_feno"], sum(mut_ind.phenotype["max_feno"]))
            
            return [ind1, mut_ind]
        
    # Выбор лучшей особи из предоставленных по значению фенотипа (можно сделать более гибким)
    def get_best_individual(self, individuals: List[Individual]) -> Individual:
        return min(individuals, key=lambda x: sum(x.phenotype['max_feno']))

    # Сортировка особей по приспособленности (значению целевой функции)
    def get_top_in_generation(self, individuals: List[Individual]) -> List[Individual]:
        result = sorted(individuals, key=lambda x: sum(x.phenotype['max_feno']))
        print(f"Отсортированные особи (всего {len(result)}): ")
        return result[:self.population_size]