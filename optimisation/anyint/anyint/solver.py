#!/usr/bin/python
# -*- coding: utf-8 -*-
import random

def solve_it(input_data):
    # return a positive integer, as a string
    random_int = random.randint(-10, 10)
    return str(random_int)

if __name__ == '__main__':
    print('This script submits the integer: %s\n' % solve_it(''))

