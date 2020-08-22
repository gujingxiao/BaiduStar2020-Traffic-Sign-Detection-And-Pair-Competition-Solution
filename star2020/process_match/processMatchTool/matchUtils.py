import numpy as np

# Get sign_id detection type
def idgettype(signid, signs_list):
    for sign_info in signs_list:
        if sign_info['sign_id'] == signid:
            return sign_info['type']

# Delete Duplicated Match for single image
def deleteDuplicatedElementFromList(list):
    resultList = []
    for item in list:
        if len(resultList) == 0:
            resultList.append(item)
        else:
            flag = 1
            for item1 in resultList:
                if item == item1:
                    flag = 0
                else:
                    continue
            if flag == 1:
                resultList.append(item)
    return resultList


def cosine_distancesmall(fea):
    dd = 0

    for index in range(len(fea[0])):
        sub_fea = []
        for idx in range(len(fea)):
            sub_fea.append(fea[idx][index])

        sub_fea = np.array(sub_fea)
        sub_fea = sub_fea.reshape(sub_fea.shape[0], -1)
        a = np.linalg.norm(sub_fea, axis=1).reshape(-1, 1)

        d = 1 - np.dot(sub_fea, sub_fea.T) / (a * a.T)
        d = d + np.eye(len(sub_fea)) * 1e8
        if index == 0:
            dd = d * 0.2
        else:
            dd = dd + d * 0.2
    return dd


def match_split(matches):
    match_a = dict()
    for match in matches:
        # if len(match['sign_id']) < 5: continue
        if match['sign_id'] in match_a:
            match_a[match['sign_id']].append(match['match_sign_id'])
        else:
            match_a[match['sign_id']] = [match['match_sign_id']]

    match_b = dict()
    for match in matches:
        # if len(match['match_sign_id']) < 5: continue
        if match['match_sign_id'] in match_b:
            match_b[match['match_sign_id']].append(match['sign_id'])
        else:
            match_b[match['match_sign_id']] = [match['sign_id']]
    return match_a, match_b


def sign2feature(signid_list, feature_signs):
    features = []
    for signid in signid_list:
        for feasign in feature_signs:
            if feasign['sign_id'] == signid:
                features.append([feasign['feature1'], feasign['feature2'], feasign['feature3'], feasign['feature4'], feasign['feature5']])
                break
    return features