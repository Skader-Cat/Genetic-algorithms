import networkx as nx
import numpy as np

from src.Evolution import Evolution

class Params:
  def set_params(self):
    self.n = int(input('Введите количество вершин n'))
    self.s_node = int(input('Введите исходящую вершину (от 0 до n):'))
    self.r1, self.r2 = [int(x) for x in input('Введите левую и правую границы генерации r1 r2').split(' ')]
    self.k = int(input('Введите количество повторов k'))
    self.n_ind = int(input('Введите количество особей в поколении Ind'))
    self.p_cross = float(input('Введите вероятность кроссовера Pk'))
    self.p_mut = float(input('Введите вероятность мутации Pm'))
    
    graph = nx.complete_graph(self.n)
    for (u, v) in graph.edges():
        graph[u][v]['weight'] = np.random.randint(self.r1, self.r2) 
    self.graph = graph
    
  def get_params(self):
    return self.n, self.graph, self.s_node, self.r1, self.r2, self.p_cross, self.p_mut, self.n_ind, self.k


def create_evolution(*params):
    evolution = Evolution(*params)
    #evolution = Evolution(7, task_matrix, 0.5, 0.5, 5, model=1)
    ostanova_counter = 0
    last_pheno = 0
    gen_counter = 0


    print("Начальное распределение:")
    evolution.print_population()
        
    generation = 0
    while True:
        print(f"Формирование нового {generation + 1}-го поколения: \n")
        print("Исходная матрица:")
        evolution.population[0].print_matrix()
        
        for current_ind in evolution.population:
            partner_ind = evolution.get_random_individual(current_ind)

            print(f"Скрещивание особей {current_ind.num} и {partner_ind.num}")
            print(f"Особь {current_ind.num}")
            current_ind.print_info()

            print(f"Особь {partner_ind.num}")
            partner_ind.print_info()

            childs = evolution.get_childs(current_ind, partner_ind)
            [child.update_pheno() for child in childs]
            best_ind = evolution.get_best_individual(childs)
            

            if evolution.model == 1:
                print(f"Лучшая особь среди потомков:")
                best_ind.print_info()
                evolution.generation_buffer[current_ind.num] = best_ind
            else:
                print(f"Получены потомки:")
                current_ind.isChild = False
                best_ind.isChild = True
                current_ind.print_info()
                best_ind.print_info()
                evolution.generation_buffer.extend([best_ind])

        print(f"Поколение {generation + 1} сформировано!")
        if evolution.model == 1:
            print(f"Лучшая особь текущего поколения:")
            best_in_gen = evolution.get_best_individual(evolution.generation_buffer.values())
            best_in_gen.print_info()
            evolution.population = list(evolution.generation_buffer.values())
        else:
            all_inds = evolution.generation_buffer
            all_inds.extend(evolution.population)
            bests_in_gen = evolution.get_top_in_generation(all_inds)
            best_in_gen = bests_in_gen[0]
            print(f"Лучшая особь из лучших:", end = "\n")
            best_in_gen.print_info()
            print(f"Все особи поколения:", end = "\n")
            [ind.print_min_info() for ind in evolution.generation_buffer]
            print(f"Лучшие особи {generation + 1} поколения:", end="\n")
            
            print("Родители и их целевые функции:", end='\n')
            evolution.print_target_func(evolution.population)
            
            print("Дети и их целевые функции:", end='\n')
            evolution.print_target_func(evolution.generation_buffer)
            
            print("Лучшие особи и их целевые функции:", end='\n')
            evolution.print_target_func(bests_in_gen)
            
            [best_child.print_min_info() for best_child in bests_in_gen]
            [best_child.mark_as_parent() for best_child in bests_in_gen]
            evolution.population = bests_in_gen
            evolution.generation_buffer = []
            
            

        if last_pheno == sum(best_in_gen.phenotype['max_feno']):
            ostanova_counter += 1
        else:
            last_pheno = sum(best_in_gen.phenotype['max_feno'])
            ostanova_counter = 0

        gen_counter += 1
        print(f"Количество поколений без изменений: {ostanova_counter + 1}")

        if ostanova_counter >= evolution.num_generations - 1:
            print("Количество поколений без изменений превысило заданное значение. Обучение завершено!")
            print("Прошло полных поколений: ", gen_counter)
            break
        
        generation += 1

    print("Обучение завершено!")