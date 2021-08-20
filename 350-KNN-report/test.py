
import math



def general_distance(fv1, fv2, p):
    # insert code here
    distance = 0.0
    for x1,x2 in zip(fv1,fv2):
        distance += math.pow(abs(x1-x2),p)
    distance = math.pow(distance, 1.0/p)
    return distance


def cosine_distance(fv1, fv2):
    # insert code here
    v1, v2, v3 = 0.0, 0.0, 0.0
    for x1,x2 in zip(fv1,fv2):
        v1 += x1*x2
        v2 += x1*x1
        v3 += x2*x2
    v2 = math.sqrt(v2)
    v3 = math.sqrt(v3)
    cosine = v1/(v2*v3)
    distance = 1 - cosine
    return distance





def KNN(train_features, train_labels, test_features, k, dist_fun, *args):
    
    predictions = []
    
    ###########################
    ## YOUR CODE BEGINS HERE
    ###########################
    def get_value(data):
        return data[0]
    for test_feature in test_features:
        distance_list = []
        for train_feature, label in zip(train_features, train_labels):
            distance = dist_fun(train_feature, test_feature, *args)
            distance_list.append([distance, label])
        distance_list.sort(key=get_value)     # 按照distance值从小到大排序
        # 统计前k次各类别出现的频数
        label_count = {}
        print(distance_list)
        for i in range(k):
            _, label = distance_list[i]
            label_count[label] = label_count.get(label, 0) + 1
        label_count = sorted(label_count.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)   # 对字典的值，从大到小排序
        predictions.append(label_count[0][0])
        print(label_count)
    ###########################
    ## YOUR CODE ENDS HERE
    ###########################
    print(predictions)
    return predictions   
    
KNN([[1,1],[5,5],[1,2]], [1,0,1], [[1,1]], 1, general_distance,2)