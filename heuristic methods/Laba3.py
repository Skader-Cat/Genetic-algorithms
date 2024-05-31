import copy
import enum
import numpy as np
import pandas as pd

class Table:
    class InputMethods(enum.Enum):
        random = 1
        from_file = 2
        manual = 3
    
    def __init__(self, input_method = None):
        self.shape = self.get_shape()
        self.low_range, self.high_range = self.get_ranges_for_random()
        if input_method == self.InputMethods.random:
            self._matrix = self.generate_random_matrix(self.shape, self.low_range, self.high_range)
        else:
            self._matrix = np.zeros(self.shape)

    def get_shape(self):
        return [int(value) for value in (input("Введите кол-во приборов N и кол-во задач M:").split(" "))]
    
    def get_ranges_for_random(self):
        return [int(value) for value in (input("Введите нижнюю и верхнюю границы значений:").split(" "))]

    def read_matrix_from_file():
        pass

    def read_matrix_from_user_input():
        pass

    def generate_random_matrix(self, shape, low_range, high_range):
        return np.random.randint(low_range, high_range, size=shape)
    
    def get_table(self, matrix):
        table = pd.DataFrame(matrix)
        table.insert(0, 'Сумма', table.sum(axis=1))
        return table
    
    #убывание
    def sort_in_desc(self):
        self._matrix = np.sort(self._matrix, axis=0)[::-1]

    #возрастание
    def sort_in_asc(self):
        self._matrix = np.sort(self._matrix, axis=0)
    

class KronMethod:
    def __init__(self, table: Table, task_vector, CMP="None"):
        if CMP == "None":
            self.table = self.initial_random_distribution(table, task_vector)
            print(self.table._matrix)
        else:
            self.table = self.find_critical_way(table, task_vector, table.shape[0])
            print(self.table._matrix)
        self._result = np.zeros(table.shape[0])
    
    def calculate(self):
        table = copy.copy(self.table)
        while True:
            sum_vector = self.get_sum_worktime_of_processes(table)
            min_sum_pos = np.argmin(sum_vector)
            max_sum_pos = np.argmax(sum_vector)

            delta = sum_vector[max_sum_pos] - sum_vector[min_sum_pos]
            
            
            if self.__get_lower_then_delta(table._matrix[max_sum_pos], delta) is not None:
                lower_then_delta = self.__get_lower_then_delta(table._matrix[max_sum_pos], delta)
                
                table._matrix[max_sum_pos][lower_then_delta[0]] = 0
                np.put(table._matrix[min_sum_pos], np.argmin(table._matrix[min_sum_pos]), lower_then_delta[1])
                continue
            else:  
                for i in range(len(table._matrix[max_sum_pos])):
                    for j in range(len(table._matrix[min_sum_pos])):
                        if table._matrix[max_sum_pos][i] - table._matrix[min_sum_pos][j] < delta:
                            if table._matrix[max_sum_pos][i] > table._matrix[min_sum_pos][j]:
                                table._matrix[max_sum_pos][i], table._matrix[min_sum_pos][j] = table._matrix[min_sum_pos][j], table._matrix[max_sum_pos][i]
                                print(table.get_table())
                                continue
                            else:
                                break
        
    def get_result(self):
        return self._result
    
    def initial_random_distribution(self, table, task_vector):
        table = copy.copy(table)
        table._matrix = np.zeros(table.shape)
        column_pos_vector = [0] * table.shape[0]
        for task in task_vector:
            #print(task, table.shape, table._matrix)
            current_row = np.random.randint(0, table.shape[0])
            current_column = column_pos_vector[current_row]
            column_pos_vector[current_row] += 1
            table._matrix[current_row][current_column] = task
            table.get_table()
        return table 
    
    def find_critical_way(self, table, vector, N_processors):
        processor_vector = [0] * N_processors
        processor_story_vector = [[] for _ in range(N_processors)]
        task_vector = vector

        table = copy.copy(table)
        table._matrix = np.zeros(table.shape)
        
        for current_task in task_vector:
            current_processor = np.argmin(processor_vector)
            processor_story_vector[current_processor].append(current_task)
            processor_vector[current_processor] += current_task
            table._matrix[current_processor][len(processor_story_vector[current_processor]) - 1] = current_task
            table.get_table()
        
        print(f"Загруженные процессоры: {processor_vector}")
        print(f"История загрузки процессоров: {processor_story_vector}")
        return table
    
    def __get_lower_then_delta(self, vector, delta):
        for i in range(len(vector)):
            if vector[i] != 0.0 and vector[i] < delta:
                return (i, vector[i])
        return None
    
    def get_sum_worktime_of_processes(self, table):
        return np.sum(table._matrix, axis=1)

"""
table = Table()
kron = KronMethod(table, [5, 10, 15, 7, 4, 22, 5])
#kron = KronMethod(table, [5, 10, 15, 7, 4, 22, 5], "CMP")

print(kron.calculate())

"""