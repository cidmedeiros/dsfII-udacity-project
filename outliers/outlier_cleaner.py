#!/usr/bin/python

def outlierCleaner(y_pred_train, x_train, y_train):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy
    from operator import itemgetter
    
    pair_up = []
    
    for i in range (len(x_train)):
        pair_up.append((x_train[i,0], y_train[i,0],
                        (numpy.power((y_pred_train[i,0]-y_train[i,0]),2))))
        
    cleaned_data = sorted(pair_up, key=itemgetter(2), reverse=True)
    c = round(len(cleaned_data) - len(cleaned_data)*0.9)
    cleaned_data = cleaned_data[c:]

    return cleaned_data

