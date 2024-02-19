from collections import Counter
def sample_accuracy(y_true, y_pred, all_or_nothing=False,mc_ml=False):
    if mc_ml == True:
        if len(y_true) != len(y_pred):
            raise ValueError("Predictions and Ground Truth must be Lists of the same length")
        count =  0
        accuracy = 0
        for x, y in zip(y_true,y_pred):
            if x != y:
                count +=1
            
        if all_or_nothing == False:
            accuracy = 1 - (count/len(y_pred))
        else: 
            if count == 1: accuracy = 0
        return accuracy
    else:
        if len(y_true) != len(y_pred):
            raise ValueError("Preidctions and Ground Truth must be lists of the same length")
    
        correct_predictions = correct_prediction(y_true=y_true,y_pred=y_pred)
        total_predictions = len(y_true)
        
        accuracy = correct_predictions/total_predictions
        return accuracy
    
def calculate_metrics():
    pass
    
    
    
def correct_prediction(y_true,y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true,y_pred)if true == pred)
    return correct_predictions

def sample_precision(y_true,y_pred,mc_ml=False,**kwargs):
    classes = set(y_true) | set(y_pred)
    macro=kwargs.get('macro',False)
    if not macro:
        #don't even use this (seriously)
        true_positives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == 1)
        true_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == 0)
        false_positiives = sum(1 for true, pred in zip(y_true,y_pred) if true == 0 and pred == 1)
        false_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == 1 and pred == 0)
        
        precision = true_positives/(true_positives+false_positiives) if (true_positives+false_positiives) else 0
        
        return precision
    precision_scores = []
    for cls in classes:
        true_positives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == cls)
        true_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred != cls)
        false_positiives = sum(1 for true, pred in zip(y_true,y_pred) if true != cls and pred == cls)
        false_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == cls and pred != cls)
        
        precision = true_positives/(true_positives+false_positiives) if (true_positives+false_positiives) else 0
        precision_scores.append(precision)
        
    return sum(precision_scores) / len(precision_scores)

def sample_recall(y_true,y_pred,mc_ml=False,**kwargs):
    classes = set(y_true) | set(y_pred)
    macro = kwargs.get('macro', False)
    if not macro:
        true_positives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == 1)
        true_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == 0)
        false_positiives = sum(1 for true, pred in zip(y_true,y_pred) if true == 0 and pred == 1)
        false_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == 1 and pred == 0)
        
        recall = true_positives/(true_positives+false_negatives) if (true_positives+false_negatives) else 0
        return recall
    recall_scores = []
    for cls in classes:
        true_positives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred == cls)
        true_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == pred != cls)
        false_positiives = sum(1 for true, pred in zip(y_true,y_pred) if true != cls and pred == cls)
        false_negatives = sum(1 for true, pred in zip(y_true,y_pred) if true == cls and pred != cls)
        
        recall = recall = true_positives/(true_positives+false_negatives) if (true_positives+false_negatives) else 0
        recall_scores.append(recall)
        
        return sum(recall_scores) / len(recall_scores)

def sample_f1_score(y_true,y_pred,mc_ml=False,**kwargs):
    macro = kwargs.get('macro',False)
    
    if not macro:
       
        precision = sample_precision(y_true=y_true,y_pred=y_pred)
        recall = sample_precision(y_pred=y_pred, y_true=y_true)
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0
        
        return f1
    classes = set(y_true) | set(y_pred)
    f1_scores = []
    for cls in classes:
        precision = sample_precision(y_true=y_true,y_pred=y_pred,macro=True)
        recall = sample_precision(y_pred=y_pred, y_true=y_true,macro=True)
        f1 = 2*(precision*recall) / (precision+recall) if (precision + recall) else 0
        f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)
    
    
def count_distribution(y_true, y_pred):
    actual_label_distributions = Counter(y_true)
    pred_label_distribution = Counter(y_pred)
    
    return actual_label_distributions,pred_label_distribution


    
        
                
    