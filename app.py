"""
● Part 1:
    ○ Merge sort, Quick sort, and Heap sort Algorithms
    ○ Insertion and Selection sort Algorithms
    ○ Time comparison Report
● Part 2:
    ○ Hybrid Merge and Insertion Algorithm
    ○ Kth Largest Element
"""
import csv
import random
from random import sample
from time import time
import pandas as pd
import matplotlib.pyplot as plt


# --------------------


def merge_sort(array: list):
    merge_sort_recursive(array, 0, len(array) - 1)


def merge_sort_recursive(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort_recursive(arr, left, mid)
        merge_sort_recursive(arr, mid + 1, right)
        merge(arr, left, mid, right)


def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid

    a = arr[0:mid + 1]
    b = arr[mid + 1:right + 1]

    for i in range(0, n1):
        a[i] = arr[left + i]

    for j in range(0, n2):
        b[j] = arr[mid + 1 + j]

    list1_index = 0
    list2_index = 0
    arr_counter = left

    while list1_index < n1 and list2_index < n2:
        if a[list1_index] >= b[list2_index]:
            arr[arr_counter] = a[list1_index]
            list1_index += 1
        else:
            arr[arr_counter] = b[list2_index]
            list2_index += 1

        arr_counter += 1

    while list1_index < n1:
        arr[arr_counter] = a[list1_index]
        list1_index += 1
        arr_counter += 1

    while list2_index < n2:
        arr[arr_counter] = b[list2_index]
        list2_index += 1
        arr_counter += 1


def find_kth_largest(array, k):
    if k < 1 or k > len(array):
        return None  # k is out of range

    return find_kth_largest_recursive(array, 0, len(array) - 1, k)


def find_kth_largest_recursive(array, left, right, k):
    pivot_index = partition(array, left, right)

    if pivot_index == k:
        return array[pivot_index]
    elif pivot_index < k:
        return find_kth_largest_recursive(array, pivot_index + 1, right, k)
    else:
        return find_kth_largest_recursive(array, left, pivot_index - 1, k)

# --------------------

def quick_sort(array):
    quick_sort_recursive(array, 0, len(array) - 1)


def quick_sort_recursive(array, left, right):
    if left < right:
        pivot = partition(array, left, right)
        quick_sort_recursive(array, left, pivot - 1)
        quick_sort_recursive(array, pivot + 1, right)


def partition(array, left, right):
    pivot_index = random.randint(left, right)
    array[left], array[pivot_index] = array[pivot_index], array[left]
    pivot = array[left]
    last_s1 = left
    first_unknown = left + 1

    while first_unknown <= right:
        if array[first_unknown] >= pivot:
            last_s1 += 1
            array[last_s1], array[first_unknown] = array[first_unknown], array[last_s1]
        first_unknown += 1

    array[left], array[last_s1] = array[last_s1], array[left]
    return last_s1 + 1


# --------------------

def heap_sort(array: list) -> list:
    build_max_heap(array)
    sorted_array = []
    while len(array) > 0:
        sorted_array.append(array[0])
        array[0], array[-1] = array[-1], array[0]
        array.pop()
        max_heapify(array, 0)
    return sorted_array


def max_heapify(array: list, index: int):
    largest = index
    left = 2 * index + 1
    right = 2 * index + 2

    if left < len(array) and array[left] > array[largest]:
        largest = left
    if right < len(array) and array[right] > array[largest]:
        largest = right

    if largest != index:
        array[index], array[largest] = array[largest], array[index]
        max_heapify(array, largest)


def build_max_heap(array: list):
    for i in range((len(array) - 1)//2, -1, -1):
        max_heapify(array, i)


# --------------------


def insertion_sort(array: list) -> list:
    if len(array) <= 1:
        return array

    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while j >= 0 and key >= array[j]:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
    return array


# --------------------

def selection_sort(array: list):
    for i in range(len(array)):
        max_element = i
        for j in range(i + 1, len(array)):
            if array[j] > array[max_element]:
                max_element = j
        array[i], array[max_element] = array[max_element], array[i]
    return array


# --------------------

def test_sorting_algorithm(problem_size):
    unsorted_array = sample(range(problem_size * 100), problem_size)
    sorting_algorithms = [merge_sort, quick_sort, heap_sort, insertion_sort, selection_sort]
    fields = ['Problem Size', 'Merge Sort', 'Quick Sort', 'Heap Sort', 'Insertion Sort', 'Selection Sort']
    filename = 'results.csv'
    row = [problem_size]
    for sorting_algorithm in sorting_algorithms:
        unsorted_array_copy = unsorted_array[:]
        start_time = time()
        sorting_algorithm(unsorted_array_copy)
        end_time = time()
        print(f'{sorting_algorithm.__name__}: {end_time - start_time}')
        row.append(end_time - start_time)
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csvwriter.writerow(fields)
        csvwriter.writerow(row)


# --------------------

def hybrid_merge_sort(array: list, k: int) -> list:
    if len(array) <= k:
        return selection_sort(array)

    first_half = hybrid_merge_sort(array[:int(len(array) / 2)], k)
    second_half = hybrid_merge_sort(array[int(len(array) / 2):], k)
    sorted_array = []

    while len(first_half) > 0 and len(second_half) > 0:
        if first_half[0] > second_half[0]:
            sorted_array.append(first_half[0])
            first_half.pop(0)

        else:
            sorted_array.append(second_half[0])
            second_half.pop(0)

    while len(first_half) > 0:
        sorted_array.append(first_half[0])
        first_half.pop(0)

    while len(second_half) > 0:
        sorted_array.append(second_half[0])
        second_half.pop(0)

    return sorted_array


# --------------------

if __name__ == '__main__':
    # test_sorting_algorithm(100)
    # test_sorting_algorithm(1000)
    # test_sorting_algorithm(25000)
    # test_sorting_algorithm(50000)
    # test_sorting_algorithm(100000)

    unsorted_array1 = sample(range(10), 10)
    print(f"before = {unsorted_array1}")
    x = find_kth_largest(unsorted_array1,3)
    #sorted_array1 = hybrid_merge_sort(unsorted_array1, 3)
    print(f"after = {x}")

    # # Read the CSV file
    # df = pd.read_csv('results.csv')
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    # for column in df.columns[1:]:
    #     plt.plot(df['Problem Size'], df[column], marker='o', label=column)
    #
    # # Add labels and title
    # plt.xlabel('Problem Size')
    # plt.ylabel('Execution Time (seconds)')
    # plt.title('Execution Time of Sorting Algorithms')
    #
    # # Add legend
    # plt.legend()
    #
    # # Save plot as PNG
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('sorting_algorithms_plot.png')  # Save plot as PNG
    # plt.show()
