import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

class Individual:
    """
    Класс, описывающий особь популяции.
    n - количество устройств
    task_vector - вектор задач
    task_matrix - матрица задач
    p_mut - вероятность мутации
    p_cross - вероятность кроссовера
    num - номер особи
    model - модель (1 - Гольдберг, 2 - Холланд)
    """
    def __init__(self, n: int, task_vector: np.ndarray, task_matrix: np.ndarray, p_mut: float, p_cross: float, num: int, model: int):
        #для моделей Холланда и Гольдберга методы одного и того же класса работают немного по-разному
        self.model = model
        self.n = n
        self.x_gens = task_vector
        #для случая, когда есть устройства, которые определенные задачи не могут считать. 
        #тогда подается матрица, состоящая из векторов задач, где -1 - это недопустимое значение
        self.full_x_gens = task_matrix
        self.x_holland = []
        self.y_gens = self.get_y_gens()
        #вероятности мутации и кроссовера
        self.mut_prop = [1] * int(p_mut * 100) + [0] * (100 - int(p_mut * 100))
        self.cross_prop = [1] * int(p_cross * 100) + [0] * (100 - int(p_cross * 100))
        self.p_mut = p_mut
        self.p_cross = p_cross
        #фенотип также отражает значение целевой функции
        self.phenotype = self.get_pheno()
        self.num = num

    def get_y_gens(self) -> List[int]:
        #если существуют устройства, которые не могут считать определенные задачи, 
        #то генотип будет генерировать такие значения, которые не позволят распределить задачу на такое устройство
        interval_size = 256 // self.n
        ranges = [(i * interval_size, (i + 1) * interval_size - 1) for i in range(self.n)]
        
        self.y_gens = []
        for row in range(self.full_x_gens.shape[0]):
            curr_indx = np.random.randint(0, self.n)
            while self.full_x_gens[row, curr_indx] == -1:
                curr_indx = np.random.randint(0, self.n)
            self.y_gens.append(np.random.randint(ranges[curr_indx][0], ranges[curr_indx][1]))
        
        return self.y_gens
          
    def print_info(self) -> None:
        print("Генотип -->")
        print("\t" + " ".join(map(str, self.x_gens if self.model == 1 else self.x_holland)))
        print("\t" + " ".join(map(str, self.y_gens)))
        self.print_pheno()
        print("\n")
  
    def print_matrix(self) -> None:
        print(pd.DataFrame(self.full_x_gens))
        
    def get_pheno(self) -> Dict[str, Union[List[Tuple[int, int]], Dict[str, List[int]], List[int], int]]:
        interval_size = 256 // self.n
        ranges = [(i * interval_size, (i + 1) * interval_size - 1) for i in range(self.n)]
        phenos = {f'{start}...{end}': [] for start, end in ranges}
        
        self.x_holland = []
        for gen_num, y_gen in enumerate(self.y_gens):
            for range_num, (start, end) in enumerate(ranges):
                if start <= y_gen < end:
                    if self.model == 1:
                        phenos[f'{start}...{end}'].append(self.x_gens[gen_num])
                    else:
                        phenos[f'{start}...{end}'].append(self.x_gens[gen_num][range_num])
                        self.x_holland.append(self.x_gens[gen_num][range_num])
        
        max_feno_num = np.argmax([sum(values) for values in phenos.values()])
        max_feno = list(phenos.values())[max_feno_num]
        
        return {
            'ranges': ranges,
            'phenos': phenos,
            'max_feno': max_feno,
            'max_feno_num': max_feno_num
        }
    
    def update_pheno(self) -> None:
        self.phenotype = self.get_pheno()

    def print_pheno(self) -> None:
        print("Фенотип:")
        for num, (pheno, values) in enumerate(self.phenotype['phenos'].items()):
            sum_values = sum(values)
            max_indicator = "<-- MAX" if num == self.phenotype['max_feno_num'] else ""
            print(f"{num + 1}) {pheno} : {values} | Сумма: {sum_values} {max_indicator}")

    def mutation(self) -> None:
        if random.random() < (self.mut_prop.count(1) / 100):
            print(f"Произошла мутация (её шанс был {self.p_mut})")
            
            interval_size = 256 // self.n
            ranges = [(i * interval_size, (i + 1) * interval_size - 1) for i in range(self.n)]
            #выбираем случайный ген для мутации
            original_gene_index = random.randint(0, len(self.y_gens) - 1)
            original_gene = self.y_gens[original_gene_index]
            mutation_attempts = 3
            success = False
            #пытаемся мутировать ген несколько раз, чтобы не попасть в недопустимое значение
            #если всё равно попадает в недопустимое, то мутация не выполняется
            for _ in range(mutation_attempts):
                random_gene_index = original_gene_index
                random_gene = original_gene

                if self.model == 1:
                    while random_gene.bit_length() == 0:
                        random_gene_index = random.randint(0, len(self.y_gens) - 1)
                        random_gene = self.y_gens[random_gene_index]

                    bit_length = random_gene.bit_length()
                    random_bit_position = random.randint(0, bit_length - 1)
                    mask = 1 << random_bit_position
                    mutated_gene = random_gene ^ mask

                else:
                    while random_gene.bit_length() < 2:
                        random_gene_index = random.randint(0, len(self.y_gens) - 1)
                        random_gene = self.y_gens[random_gene_index]

                    bit_length = random_gene.bit_length()
                    random_bit_positions = random.sample(range(bit_length), 2)
                    mask1 = 1 << random_bit_positions[0]
                    mask2 = 1 << random_bit_positions[1]
                    mutated_gene = random_gene ^ mask1 ^ mask2

                interval_number = next(i for i, (start, end) in enumerate(ranges) if start <= mutated_gene <= end)
                value = self.full_x_gens[random_gene_index][interval_number]
                print(f"Мутировал ген c индексом {random_gene_index} с {random_gene} на {mutated_gene} (значение {value})")
                
                if value == -1:
                    print(f"Мутация №{_ + 1} не удалась, ген мутировал в недопустимое значение")
                    print(f"Позиция в исходной матрице: {random_gene_index} {interval_number}")
                    print("Исходная матрица:")
                    self.print_matrix()
                else:
                    self.y_gens[random_gene_index] = mutated_gene
                    self.update_pheno()
                    success = True
                    break

            if not success:
                self.y_gens[random_gene_index] = original_gene
                print("Мутация не удалась после 3 попыток, ген остался исходным")
        else:
            print("Мутация не произошла")

    def crossover(self, ind: 'Individual') -> Optional[List['Individual']]:
        if random.random() < self.p_cross:
            first_child = Individual(ind.n, ind.x_gens, ind.full_x_gens, ind.p_mut, ind.p_cross, num=self.num, model=self.model)
            second_child = Individual(ind.n, ind.x_gens, ind.full_x_gens, ind.p_mut, ind.p_cross, num=self.num, model=self.model)
            #Одноточечный кроссовер по Гольдбергу
            if self.model == 1:
                split_dot = np.random.randint(1, len(self.y_gens) - 1)
                first_child.y_gens = self.y_gens[:split_dot] + ind.y_gens[split_dot:]
                second_child.y_gens = ind.y_gens[:split_dot] + self.y_gens[split_dot:]
            #Двухточечный по Холланду
            else:
                print("По модели Холланда")
                left_split_dot, right_split_dot = sorted(np.random.randint(1, len(self.y_gens) - 1, 2))
                
                first_child.y_gens = self.y_gens[:left_split_dot] + ind.y_gens[left_split_dot:right_split_dot] + self.y_gens[right_split_dot:]
                second_child.y_gens = ind.y_gens[:left_split_dot] + self.y_gens[left_split_dot:right_split_dot] + ind.y_gens[right_split_dot:]

            print(f"Прошёл кроссовер (его шанс был {self.p_cross})")
            
            return [first_child, second_child]
        else:
            print("Кроссовер не произошёл")
            return None