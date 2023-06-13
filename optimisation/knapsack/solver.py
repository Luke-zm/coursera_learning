#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def trivial_sol(val, weight, taken, items, capacity):
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            val += item.value
            weight += item.weight
    return val, taken

def dynamic_programming(taken, items, capacity):
    # Using Dynamic Programming to solve the problem
    # Filter the problem column by column

    # Find the number of rows of the table, where num_rows = capacity + 1
    num_rows = capacity + 1
    # Finf the number of colmuns of the table, where num_col = num_item + 1
    num_col = int(len(items)) + 1
    # Build a table of the dynamic optimisation problem
    table = np.zeros((num_rows, num_col), dtype= int)

    # prepare the items as lists for easier manipulation
    wt = []
    val = []
    for item in items:
        wt.append(item.weight)
        val.append(item.value)

    # sequence is found by checking from btm right corner
    for item in range(num_col):
        for weight_idx in range(num_rows):
            if item == 0 or weight_idx == 0:
                # for base case of weight = 0 and item = 0
                table[weight_idx][item] = 0
            elif wt[item-1] <= weight_idx:
                # if the same weight is allowed, select the larger value by comparing
                # current item + all previous taken items, and
                # all previous taken items
                table[weight_idx][item] = max(val[item - 1] + table[weight_idx - wt[item - 1]][item - 1],
                                              table[weight_idx][item - 1])
            else:
                table[weight_idx][item] = table[weight_idx][item - 1]
    # value is the number on the btm right corner
    result = table[-1][-1]
    final_val = result

    # find what items are taken
    w = capacity
    for i in range(num_col, 0, -1):
        if result <= 0:
            break
        # either the result comes from the
        # top (table[w][i-1]) or from (val[i-1]
        # + table[w-wt[i-1]] [i-1]) as in Knapsack
        # table. If it comes from the latter
        # one/ it means the item is included.
        if result == table[w][i - 1]:
            continue
        else:
 
            # This item is included.
            taken[i-1] = 1
             
            # Since this weight is included
            # its value is deducted
            result = result - val[i - 1]
            w = w - wt[i - 1]

    return final_val, taken

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)
    print(capacity)
    print((capacity >= 1000000))
    if (capacity >= 1000000):
        print("too large")
        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
        value, taken = trivial_sol(value, weight, taken, items, capacity)
    else:
        value, taken = dynamic_programming(taken, items, capacity)
    
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

