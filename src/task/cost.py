def get_cost_func(cost_func_name):
    if cost_func_name == 'L-distance':
        return lambda correction: correction.edit_distance()
    else:
        raise ValueError(f'Unknown cost function {cost_func_name}')