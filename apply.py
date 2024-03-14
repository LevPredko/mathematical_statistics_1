import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def read_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            numbers = list(map(int, file.read().split(',')))
            return numbers
    except FileNotFoundError:
        print(f"Файл '{file_path}' не знайдено.")
    except Exception as e:
        print(f"Виникла помилка при читанні файлу: {e}")


def variation_series(file_path): # ВАРІАЦІЙНИЙ РЯД ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = read_from_file(file_path)
    return sorted(numbers)


def generate_midpoint_array(file_path): # ВАРІАЦІЙНИЙ РЯД ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    bin_edges, hist = continuous_frequency_table(file_path, print_table=False)

    midpoint_array = []
    for i in range(len(hist)):
        midpoint = (bin_edges[i] + bin_edges[i + 1]) / 2
        frequency = hist[i]
        midpoint_array.extend([midpoint] * frequency)

    return midpoint_array



def swing_discrete(file_path): # РОЗМАХ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    minValue = min(numbers)
    maxValue = max(numbers)
    return maxValue - minValue


def swing_continuouse(file_path): # РОЗМАХ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    minValue = min(numbers)
    maxValue = max(numbers)
    return maxValue - minValue


def average_discrete(file_path): # СЕРЕДНЄ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    average_num = sum(numbers) / len(numbers)
    return average_num


def average_continuouse(file_path): # СЕРЕДНЄ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    average_num = sum(numbers) / len(numbers)
    return average_num


def moda_discrete(file_path): # МОДА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    counts = Counter(numbers)
    most_common, occurrences = counts.most_common(1)[0]
    return most_common


def moda_continuouse(file_path): # МОДА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    counts = Counter(numbers)
    most_common, occurrences = counts.most_common(1)[0]
    return most_common


def mediana_discrete(file_path): # МЕДІАНА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    if len(numbers) % 2 == 0:
        middle1 = numbers[len(numbers) // 2]
        middle2 = numbers[len(numbers) // 2 - 1]
        return (middle1 + middle2) / 2
    else:
        return numbers[len(numbers) // 2]


def mediana_continuouse(file_path): # МЕДІАНА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    if len(numbers) % 2 == 0:
        middle1 = numbers[len(numbers) // 2]
        middle2 = numbers[len(numbers) // 2 - 1]
        return ((middle1 + middle2) / 2)
    else:
        return numbers[len(numbers) // 2]


def variance_discrete(file_path): # ВАРІАНСА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    return deviation_discrete(file_path) / (len(numbers) - 1)


def variance_continuouse(file_path): # ВАРІАНСА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    return deviation_continuouse(file_path) / (len(numbers) - 1)


def deviation_discrete(file_path): # ДЕВІАЦІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    average_num = average_discrete(file_path)
    deviation = [(x - average_num) ** 2 for x in numbers]
    deviations_value = sum(deviation)
    return deviations_value


def deviation_continuouse(file_path): # ДЕВІАЦІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    average_num = average_continuouse(file_path)
    deviation = [(x - average_num) ** 2 for x in numbers]
    deviations_value = sum(deviation)
    return deviations_value


def dispersion_discrete(file_path): # ДИСПЕРСІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    return deviation_discrete(file_path) / (len(numbers))


def dispersion_continuouse(file_path): # ДИСПЕРСІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    return deviation_continuouse(file_path) / (len(numbers))


def mean_square_deviation_discrete(file_path): # СЕРЕДНЄ КВАДРАТИЧНЕ ВІДХИЛЕННЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    return math.sqrt(dispersion_discrete(file_path))


def mean_square_deviation_continuouse(file_path): # СЕРЕДНЄ КВАДРАТИЧНЕ ВІДХИЛЕННЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    return math.sqrt(dispersion_continuouse(file_path))


def standard_discrete(file_path): # СТАНДАРТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    return math.sqrt(variance_discrete(file_path))


def standard_continuouse(file_path): # СТАНДАРТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    return math.sqrt(variance_continuouse(file_path))


def variation_discrete(file_path): # ВАРІАЦІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    return standard_discrete(file_path)/average_discrete(file_path)


def variation_continuouse(file_path): # ВАРІАЦІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    return standard_continuouse(file_path)/average_continuouse(file_path)


def quantile_discrete(file_path, q): # КВАНТИЛЬ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    index = q * (len(numbers) / 100)

    if index.is_integer():
        return numbers[int(index)]
    else:
        return


def quantile_continuouse(file_path, q): # КВАНТИЛЬ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    index = q * (len(numbers) / 100)

    if index.is_integer():
        return numbers[int(index)]
    else:
        return


def quartiles_discrete(file_path): # КВАРТИЛЬ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    q1 = quantile_discrete(file_path, 25)
    q2 = quantile_discrete(file_path, 50)
    q3 = quantile_discrete(file_path, 75)

    return q1, q2, q3

def quartiles_continuouse(file_path): # КВАРТИЛЬ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    q1 = quantile_continuouse(file_path, 25)
    q2 = quantile_continuouse(file_path, 50)
    q3 = quantile_continuouse(file_path, 75)

    return q1, q2, q3


def interquartile_range_discrete(file_path): # ІНТЕРКВАРТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    q1 = quantile_discrete(file_path, 25)
    q3 = quantile_discrete(file_path, 75)
    try:
        ior = q3 - q1
    except(Exception):
        ior = None

    return ior



def interquartile_range_continuous(file_path): # ІНТЕРКВАРТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    q1 = quantile_continuouse(file_path, 25)
    q3 = quantile_continuouse(file_path, 75)
    try:
        ior = q3 - q1
    except(Exception):
        ior = None

    return ior


def octile_discrete(file_path): # ОКТИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    o1 = quantile_discrete(file_path, 12.5)
    o2 = quantile_discrete(file_path, 25)
    o3 = quantile_discrete(file_path, 37.5)
    o4 = quantile_discrete(file_path, 50)
    o5 = quantile_discrete(file_path, 62.5)
    o6 = quantile_discrete(file_path, 75)
    o7 = quantile_discrete(file_path, 87.5)
    return o1, o2, o3, o4, o5, o6, o7


def octile_continuouse(file_path): # ОКТИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    o1 = quantile_continuouse(file_path, 12.5)
    o2 = quantile_continuouse(file_path, 25)
    o3 = quantile_continuouse(file_path, 37.5)
    o4 = quantile_continuouse(file_path, 50)
    o5 = quantile_continuouse(file_path, 62.5)
    o6 = quantile_continuouse(file_path, 75)
    o7 = quantile_continuouse(file_path, 87.5)
    return o1, o2, o3, o4, o5, o6, o7


def interoctile_range_discrete(file_path): # ІНТЕРОКТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    o1 = quantile_discrete(file_path, 12.5)
    o7 = quantile_discrete(file_path, 87.5)
    try:
        ior = o7 - o1
    except(Exception ):
        ior = None

    return ior


def interoctile_range_continuous(file_path): # ІНТЕРОКТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    o1 = quantile_continuouse(file_path, 12.5)
    o7 = quantile_continuouse(file_path, 87.5)
    try:
        ior = o7 - o1
    except(Exception):
        ior = None

    return ior


def decile_discrete(file_path): # ДЕЦИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    d1 = quantile_discrete(file_path, 10)
    d2 = quantile_discrete(file_path, 20)
    d3 = quantile_discrete(file_path, 30)
    d4 = quantile_discrete(file_path, 40)
    d5 = quantile_discrete(file_path, 50)
    d6 = quantile_discrete(file_path, 60)
    d7 = quantile_discrete(file_path, 70)
    d8 = quantile_discrete(file_path, 80)
    d9 = quantile_discrete(file_path, 90)
    return d1, d2, d3, d4, d5, d6, d7, d8, d9


def decile_continuouse(file_path): # ДЕЦИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    d1 = quantile_continuouse(file_path, 10)
    d2 = quantile_continuouse(file_path, 20)
    d3 = quantile_continuouse(file_path, 30)
    d4 = quantile_continuouse(file_path, 40)
    d5 = quantile_continuouse(file_path, 50)
    d6 = quantile_continuouse(file_path, 60)
    d7 = quantile_continuouse(file_path, 70)
    d8 = quantile_continuouse(file_path, 80)
    d9 = quantile_continuouse(file_path, 90)
    return d1, d2, d3, d4, d5, d6, d7, d8, d9


def interdecile_range_discrete(file_path): # ІНТЕРДЕЦИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    d1 = quantile_discrete(file_path, 10)
    d9 = quantile_discrete(file_path, 90)
    try:
        idr = d9 - d1
    except( Exception):
        idr = None
    return idr

def interdecile_range_continuous(file_path):  # ІНТЕРДЕЦИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    d1 = quantile_continuouse(file_path, 10)
    d9 = quantile_continuouse(file_path, 90)
    idr = d9 - d1
    return idr


def centile_discrete(file_path): # ЦЕНТИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    centiles = []
    for i in range(1, 100):
        centile = quantile_discrete(file_path, i)
        centiles.append(centile)
    return centiles


def centile_continuouse(file_path): # ЦЕНТИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    centiles = []
    for i in range(1, 100):
        centile = quantile_continuouse(file_path, i)
        centiles.append(centile)
    return centiles


def intercentile_range_discrete(file_path): # ІНТЕРЦЕНТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    c1 = quantile_discrete(file_path, 1)
    c99 = quantile_discrete(file_path, 99)
    try:
        ior = c99 - c1
    except(Exception):
        ior = None

    return ior


def intercentile_range_continuous(file_path): # ІНТЕРЦЕНТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    c1 = quantile_continuouse(file_path, 1)
    c99 = quantile_continuouse(file_path, 99)
    try:
        ior = c99 - c1
    except(Exception):
        ior = None

    return ior


def millesile_discrete(file_path): # МІЛІЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    millesiles = []
    for i in range(1, 1000):
        millesile = quantile_discrete(file_path, i/10)
        millesiles.append(millesile)
    return millesiles


def millesile_continuouse(file_path): # МІЛІЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    millesiles = []
    for i in range(1, 1000):
        millesile = quantile_continuouse(file_path, i/10)
        millesiles.append(millesile)
    return millesiles


def intermillesile_range_discrete(file_path): # ІНТЕРМІЛІЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    m1 = quantile_discrete(file_path, 1)
    m1000 = quantile_discrete(file_path, 999)
    try:
        ior = m1000 - m1
    except(Exception):
        ior = None

    return ior


def intermillesile_range_continuous(file_path): # ІНТЕРМІЛІЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    m1 = quantile_continuouse(file_path, 1)
    m1000 = quantile_continuouse(file_path, 999)
    try:
        ior = m1000 - m1
    except(Exception):
        ior = None

    return ior


def moment_discrete(file_path, x, k): # МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    my_list = []
    for element in numbers:
        my_list.append((element - x)**k)
    return 1/len(numbers) * sum(my_list)


def moment_continuouse(file_path, x, k): # МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = generate_midpoint_array(file_path)
    my_list = []
    for element in numbers:
        my_list.append((element - x)**k)
    return 1/len(numbers) * sum(my_list)


def asymmetry_discrete(file_path): # АСИМЕТРІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    return (moment_discrete(file_path, average_discrete(file_path), 3) / (moment_discrete(file_path, average_discrete(file_path), 2))**(3/2))


def asymmetry_continuouse(file_path): # АСИМЕТРІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    return (moment_continuouse(file_path, average_continuouse(file_path), 3) / (moment_continuouse(file_path, average_continuouse(file_path), 2))**(3/2))


def excess_discrete(file_path): # ЕКСЦЕС ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    return (moment_discrete(file_path, average_discrete(file_path), 4) / (moment_discrete(file_path, average_discrete(file_path), 2))**(3 / 2))


def excess_continuouse(file_path): # ЕКСЦЕС ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    return (moment_continuouse(file_path, average_continuouse(file_path), 4) / (moment_continuouse(file_path, average_continuouse(file_path), 2))**(3 / 2))


def discrete_frequency_table(file_path): # ТАБЛИЦЯ ЧАСТОТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    counts = Counter(numbers)
    print("\nТАБЛИЦЯ ЧАСТОТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ:")
    print("╔══════════╦═════════╗")
    print("║ Значення ║ Частота ║")
    print("╠══════════╬═════════╣")
    for index, (value, frequency) in enumerate(counts.items()):
        print(f"║ {value:^9}║ {frequency:^8}║")
        if index < len(counts) - 1:
            print("╠══════════╬═════════╣")
    print("╚══════════╩═════════╝")



def calculate_num_bins(numbers): # КІЛЬКІСТЬ ВІДРІЗКІВ В ТАБЛИЦІ ЧАСТОТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    p = np.max(numbers) - np.min(numbers)
    r = int(np.ceil(np.log2(len(numbers)) - 1))
    return r


def continuous_frequency_table(file_path, print_table=True): #ТАБЛИЦЯ ЧАСТОТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ
    numbers = variation_series(file_path)
    num_bins = calculate_num_bins(numbers)
    bin_edges = np.linspace(-15, 15, num_bins+1)
    hist, _ = np.histogram(numbers, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if(print_table):
        print("\nТАБЛИЦЯ ЧАСТОТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ:")
        print("╔═══════════════════╦═══════════════╦══════════╗")
        print("║    Відрізок       ║   Середина    ║ Частота  ║")
        print("╠═══════════════════╬═══════════════╬══════════╣")
        for i in range(len(hist)):
            print(f"║ {bin_edges[i]:^8.2f}-{bin_edges[i + 1]:^8.2f} ║ {bin_centers[i]:^13.2f} ║ {hist[i]:^7}  ║")
            if(i != len(hist) - 1):
                print("╠═══════════════════╬═══════════════╬══════════╣")
        print("╚═══════════════════╩═══════════════╩══════════╝")

    return bin_edges, hist


def cumulative_distribution_function_discrete(file_path, print_table=True): #ФУНКЦІОНАЛЬНИЙ РОЗПОДІЛ
    numbers = variation_series(file_path)
    counts = Counter(numbers)
    total_count = len(numbers)

    cumulative_probabilities = []
    cumulative_probability = 0

    if print_table:
        print("\nФУНКЦІОНАЛЬНИЙ РОЗПОДІЛ:")
        print("╔══════════╦═══════════════════════════╗")
        print("║ Значення ║ Кумулятивна ймовірність   ║")
        print("╠══════════╬═══════════════════════════╣")


    for value, frequency in sorted(counts.items()):
        probability = frequency / total_count
        cumulative_probability += probability
        cumulative_probabilities.append((value, cumulative_probability))


    if print_table:

        for i, (value, cumulative_probability) in enumerate(cumulative_probabilities):
            if i == 0:
                lower_bound = "-ထ"
            else:
                lower_bound = cumulative_probabilities[i][0]

            if i == len(cumulative_probabilities) - 1:
                upper_bound = "ထ"
                sign = "<"
            else:
                upper_bound = value
                sign = "≤"


            x = "x"
            print(f"║ {cumulative_probability:^8.2f} ║ {lower_bound:^7} < {x:^6}{sign}{upper_bound:^7} ║")

        print("╚══════════╩═══════════════════════════╝")




def plot_discrete_polygon(file_path): #ПОЛІГОН ЧАСТОТ
    numbers = variation_series(file_path)
    counts = Counter(numbers)
    values, frequencies = zip(*sorted(counts.items()))

    plt.figure(figsize=(10, 6))
    plt.plot(values, frequencies, marker='o', linestyle='-')
    plt.title('Полігон Частот')
    plt.xlabel('Значення')
    plt.ylabel('Частоти')
    plt.grid(True)
    plt.show()


def plot_absolute_frequency_discrete(file_path): # ДІАГРАМА АБСОЛЮТНИХ ЧАСТОТ
    numbers = variation_series(file_path)
    counts = Counter(numbers)

    values = list(counts.keys())
    frequencies = list(counts.values())

    plt.bar(values, frequencies, color='blue', alpha=0.9)
    plt.xlabel('Значення')
    plt.ylabel('Абсолютна частота')
    plt.title('Діаграма абсолютних частот')
    plt.show()


def plot_cumulative_distribution(cumulative_probabilities): #ФУНКЦІОНАЛЬНИЙ РОЗПОДІЛ
    values, probabilities = zip(*cumulative_probabilities)

    plt.figure(figsize=(8, 6))

    for i in range(len(values) - 1):
        if values[i] != -np.inf and values[i + 1] != np.inf:
            plt.plot([values[i], values[i + 1]], [probabilities[i], probabilities[i]], 'b-')
            plt.plot(values[i], probabilities[i], 'ro', markersize=4)
        else:
            plt.plot([values[i], values[i + 1]], [probabilities[i], probabilities[i]], 'b-')


    plt.plot([-20, -15], [0, 0], 'b-')
    plt.plot([15, 15], [0, 1], 'ro', markersize=4)
    plt.plot([15, 20], [1, 1], 'b-')

    plt.xlim([-20, 20])

    plt.xlabel('Значення')
    plt.ylabel('Кумулятивна ймовірність')
    plt.title('Функціональний розподіл')

    plt.grid(True)
    plt.show()


def plot_continuous_frequency_histogram(bin_edges, hist): #ГІСТОГРАМА
    plt.bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], color='blue', edgecolor='black', align='edge')
    plt.title("Гістограма для неперервної змінної")
    plt.xlabel("Відрізок")
    plt.ylabel("Частота")
    plt.xlim(-15, 15)
    plt.show()


if __name__ == '__main__':

   FILE = "/Users/lev/Documents/programming/mathematical_statistics_1/Vibirka"
   print("ВИБІРКА: ",read_from_file("/Users/lev/Documents/programming/mathematical_statistics_1/Vibirka"))
   print("ВАРІАЦІЙНИЙ РЯД ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",variation_series(FILE))
   print("ДОВЖИНА ВАРІАЦІЙНОГО РЯДУ:",len(variation_series(FILE)))
   print("СЕРЕДНЄ ЗНАЧЧЕННЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",average_discrete(FILE))
   print("РОЗМАХ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", swing_discrete(FILE))
   print("МОДА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moda_discrete(FILE))
   print("МЕДІАНА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", mediana_discrete(FILE))
   print("ВАРІАНСА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",variance_discrete(FILE))
   print("ДЕВІАЦІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",deviation_discrete(FILE))
   print("ДИСПЕРСІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",dispersion_discrete(FILE))
   print("СЕРЕДНЄ КВАДРАТИЧНЕ ВІДХИЛЕННЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",mean_square_deviation_discrete(FILE))
   print("СТАНДАРТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",standard_discrete(FILE))
   print("ВАРІАЦІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",variation_discrete(FILE))
   print("КВАНТИЛЬ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ","\n\t\t\tQuantile at 0.3:", quantile_discrete(FILE, 0.3))
   print("КВАРТИЛЬ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ", "\n\t\t\tFirst, second (median), and third quartiles:", quartiles_discrete(FILE))
   print("ІНТЕРКВАРТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", interquartile_range_discrete(FILE))
   print("ОКТИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",octile_discrete(FILE))
   print("ІНТЕРОКТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",interoctile_range_discrete(FILE))
   print("ДЕЦИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",decile_discrete(FILE))
   print("ІНТЕРДЕЦИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", interdecile_range_discrete(FILE))
   print("ЦЕНТИЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",centile_discrete(FILE))
   print("ІНТЕРЦЕНТИЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", intercentile_range_discrete(FILE))
   print("МІЛІЛІ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ",millesile_discrete(FILE))
   print("ІНТЕРМІЛІЛЬНА ШИРОТА ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", intermillesile_range_discrete(FILE))
   print("ПЕРШИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE,  0, 1))
   print("ДРУГИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, 0, 2))
   print("ТРЕТІЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, 0, 3))
   print("ЧЕТВЕРТИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, 0, 4))
   print("ПЕРШИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, average_discrete(FILE), 1))
   print("ДРУГИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, average_discrete(FILE), 2))
   print("ТРЕТІЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, average_discrete(FILE), 3))
   print("ЧЕТВЕРТИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", moment_discrete(FILE, average_discrete(FILE), 4))
   print("АСИМЕТРІЯ ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", asymmetry_discrete(FILE))
   print("ЕКСЦЕС ДЛЯ ДИСКРЕТНОЇ ЗМІННОЇ: ", excess_discrete(FILE))
   discrete_frequency_table(FILE)
   cumulative_distribution_function_discrete(FILE)


   print("\nВАРІАЦІЙНИЙ РЯД ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", generate_midpoint_array(FILE))
   print("СЕРЕДНЄ ЗНАЧЧЕННЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", average_continuouse(FILE))
   print("РОЗМАХ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", swing_continuouse(FILE))
   print("МОДА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", moda_continuouse(FILE))
   print("МЕДІАНА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", round(mediana_continuouse(FILE)))
   print("ВАРІАНСА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", variance_continuouse(FILE))
   print("ДЕВІАЦІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", deviation_continuouse(FILE))
   print("ДИСПЕРСІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", dispersion_continuouse(FILE))
   print("СЕРЕДНЄ КВАДРАТИЧНЕ ВІДХИЛЕННЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", mean_square_deviation_continuouse(FILE))
   print("СТАНДАРТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", standard_continuouse(FILE))
   print("ВАРІАЦІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", variation_continuouse(FILE))
   print("КВАНТИЛЬ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", "\n\t\t\tQuantile at 0.3:", quantile_continuouse(FILE, 0.3))
   print("КВАРТИЛЬ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", "\n\t\t\tFirst, second (median), and third quartiles:", quartiles_continuouse(FILE))
   print("ІНТЕРКВАРТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", interquartile_range_continuous(FILE))
   print("ОКТИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", octile_continuouse(FILE))
   print("ІНТЕРОКТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", interoctile_range_continuous(FILE))
   print("ДЕЦИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", decile_continuouse(FILE))
   print("ІНТЕРДЕЦИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", interdecile_range_continuous(FILE))
   print("ЦЕНТИЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", centile_continuouse(FILE))
   print("ІНТЕРЦЕНТИЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", intercentile_range_continuous(FILE))
   print("МІЛІЛІ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", millesile_continuouse(FILE))
   print("ІНТЕРМІЛІЛЬНА ШИРОТА ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", intermillesile_range_continuous(FILE))
   print("ПЕРШИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, 0,1))
   print("ДРУГИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, 0,2))
   print("ТРЕТІЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, 0,3))
   print("ЧЕТВЕРТИЙ ПОЧАТКОВИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, 0,4))
   print("ПЕРШИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", moment_continuouse(FILE, average_continuouse(FILE),1))
   print("ДРУГИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, average_continuouse(FILE),2))
   print("ТРЕТІЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, average_continuouse(FILE),3))
   print("ЧЕТВЕРТИЙ ЦЕНТРАЛЬНИЙ МОМЕНТ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ",moment_continuouse(FILE, average_continuouse(FILE),4))
   print("АСИМЕТРІЯ ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", asymmetry_continuouse(FILE))
   print("ЕКСЦЕС ДЛЯ НЕПЕРЕРВНОЇ ЗМІННОЇ: ", excess_continuouse(FILE))
   continuous_frequency_table(FILE)

   plot_discrete_polygon(FILE) #ПОЛІГОН ЧАСТОТ

   plot_cumulative_distribution(cumulative_distribution_function_discrete(FILE,False))#ФУНКЦІОНАЛЬНИЙ РОЗПОДІЛ

   bin_edges, hist = continuous_frequency_table(FILE, False) #ГІСТОГРАМА
   plot_continuous_frequency_histogram(bin_edges, hist)

   plot_absolute_frequency_discrete(FILE) # ДІАГРАМА АБСОЛЮТНИХ ЧАСТОТ
