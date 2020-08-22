import cv2
import numpy as np
from .matchUtils import idgettype, deleteDuplicatedElementFromList, cosine_distancesmall, sign2feature
from .SiftMatch import getSiftMatchScore

labels = {
    '102': 0,
    '103': 1,
    '104': 2,
    '105': 3,
    '106': 4,
    '107': 5,
    '108': 6,
    '109': 7,
    '110': 8,
    '111': 9,
    '112': 10,
    '201': 11,
    '202': 12,
    '203': 13,
    '204': 14,
    '205': 15,
    '206': 16,
    '207': 17,
    '301': 18,
}

classes = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111',
           '112', '201', '202', '203', '204', '205', '206', '207', '301']

def correctWrongOrder(preMatchs):
    pdMatchsNew = []

    # Delete or Fix Wrong Order
	# 将顺序不对或者异常的进行删除或重排序
    for pdMatch in preMatchs:
        pdSign_id = pdMatch['sign_id']
        pdMatch_sign_id = pdMatch['match_sign_id']
        pdMatch_score = pdMatch['match_score']

        if 'B' in pdSign_id and pdMatch_sign_id == " ":
            continue
        elif 'A' in pdSign_id and 'A' in pdMatch_sign_id:
            continue
        elif 'B' in pdSign_id:
            pdMatchsNew.append({'sign_id': pdMatch_sign_id, 'match_sign_id': pdSign_id, 'match_score': pdMatch_score})
        else:
            pdMatchsNew.append(pdMatch)

    return pdMatchsNew


def deleteWrongTypeMatch(pdMatchs, preSigns):
    # Delete Wrong Type
	# 将两个目标类型不同的进行删除
    filter_match = []
    for match_info in pdMatchs:
        sign_a = match_info['sign_id']
        sign_b = match_info['match_sign_id']
        sign_a_type = idgettype(sign_a, preSigns)
        sign_b_type = idgettype(sign_b, preSigns)
        sign_score = match_info['match_score']

        new_match = dict()
        if sign_a_type == sign_b_type:
            new_match['sign_id'] = sign_a
            new_match['match_sign_id'] = sign_b
            new_match['match_score'] = sign_score
            filter_match.append(new_match)

    for match_info in pdMatchs:
        sign_a = match_info['sign_id']
        sign_b = match_info['match_sign_id']
        sign_a_type = idgettype(sign_a, preSigns)
        sign_b_type = idgettype(sign_b, preSigns)
        sign_score = match_info['match_score']

        new_match = dict()
        if sign_a_type != sign_b_type:
            ids = []
            for match in filter_match:
                ids.append(match['sign_id'])
            if sign_a not in ids:
                new_match['sign_id'] = sign_a
                new_match['match_sign_id'] = " "
                new_match['match_score'] = sign_score
                filter_match.append(new_match)
	# Check duplicated items and delete them
	# 检查一下元素并将重复的删除
    simplifiedPdMatchs = deleteDuplicatedElementFromList(filter_match)
    return simplifiedPdMatchs


def filterThresholdMatch(pdMatchs, preSigns, det_thres, mt_thres):
    thresPdSigns = []
    thresPdMatchs = []
    sign_id_list = []
	# Filter Match by detection threshold
	# 通过检测阈值来过滤检测目标和匹配目标
    for pdSign in preSigns:
        if 'score' not in pdSign:
            thresPdSigns.append(pdSign)
            sign_id_list.append(pdSign['sign_id'])
        else:
            if pdSign['score'] >= det_thres[labels[pdSign['type']]]:
                thresPdSigns.append(pdSign)
                sign_id_list.append(pdSign['sign_id'])
    preSigns = thresPdSigns
    sign_id_list.append(" ")

    for pdMatch in pdMatchs:
        if pdMatch['sign_id'] in sign_id_list and pdMatch['match_sign_id'] in sign_id_list:
            signtype = '301'
            for pdSign in preSigns:
                if pdMatch['sign_id'] == pdSign['sign_id']:
                    signtype = pdSign['type']

            if pdMatch['match_score'] < mt_thres[labels[signtype]]:
                thresPdMatchs.append(pdMatch)

    return thresPdMatchs, preSigns


def deleteUnreasonableMatch(preJson, imageResolutions):
    # Delete Some Unreasonable Matchs
	# 根据位置信息删除一些毫无依据的匹配目标
    filterMatchs = []
    allInfoMatchs = []
    for match_info in preJson['match']:
        sign_a = match_info['sign_id']
        sign_b = match_info['match_sign_id']

        gtBox0 = []
        gtBox1 = []
        pic0 = ''
        pic1 = ''
        type0 = ''
        type1 = ''
        for sign_info in preJson['signs']:
            if sign_info['sign_id'] == sign_a:
                pic0 = sign_info['pic_id']
                type0 = sign_info['type']
                gtBox0 = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]

            if sign_info['sign_id'] == sign_b:
                pic1 = sign_info['pic_id']
                type1 = sign_info['type']
                gtBox1 = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]

        if gtBox1 == []:
            filterMatchs.append(match_info)
            allInfo = match_info
            allInfo['sign_pic'] = pic0
            allInfo['match_pic'] = pic1
            allInfo['sign_type'] = type0
            allInfo['match_type'] = type1
            allInfoMatchs.append(allInfo)
        else:
            w0 = imageResolutions[pic0][0]
            w1 = imageResolutions[pic1][0]

            if (gtBox0[2] < w0 * 0.3 and gtBox1[0] > w1 * 0.7) or (gtBox0[2] < w0 * 0.15 and gtBox1[0] > w1 * 0.55):
                continue
            elif (gtBox0[0] > w0 * 0.7 and gtBox1[2] < w1 * 0.3) or (gtBox0[0] > w0 * 0.55 and gtBox1[2] < w1 * 0.15):
                continue
            elif ((gtBox0[0] > w0 * 0.6 and gtBox1[2] < w1 * 0.4) or (gtBox0[0] > w0 * 0.5 and gtBox1[2] < w1 * 0.2)) and type0 == '301':
                continue
            else:
                filterMatchs.append(match_info)
                allInfo = match_info
                allInfo['sign_pic'] = pic0
                allInfo['match_pic'] = pic1
                allInfo['sign_type'] = type0
                allInfo['match_type'] = type1
                allInfoMatchs.append(allInfo)
    return filterMatchs, allInfoMatchs

def filterMultiToOneMatch(preJson, allInfoMatchs, postMethod, picFolder, confidenceRange):
    # Filter multi to one match
	# 针对同一张图不同目标匹配同一个的情况，进行过滤处理
    haveMatchs = []
    noMatchs = []
    tmpMatchs = []
    for allInfo in allInfoMatchs:
        if allInfo['match_sign_id'] == '' or allInfo['match_sign_id'] == ' ':
            tmpMatchs.append(allInfo)
        else:
            haveMatchs.append(allInfo)

    for noMatch in tmpMatchs:
        flag = 0
        for haveMatch in haveMatchs:
            if noMatch['sign_id'] == haveMatch['sign_id']:
                flag = 1
                print(flag)
        if flag == 0:
            noMatchs.append(noMatch)

	# Get multi-to-one pairs and collect them
	# 寻找多对一的匹配对
    singleMatchs = []
    duplicatedMatchs = []
    duplicatedList = []
    for sub in range(len(haveMatchs)):
        singleFlag = 0
        for sub_idd in range(len(haveMatchs)):
            if haveMatchs[sub]['sign_pic'] == haveMatchs[sub_idd]['sign_pic'] and \
                haveMatchs[sub]['match_pic'] == haveMatchs[sub_idd]['match_pic'] and \
                haveMatchs[sub]['match_sign_id'] == haveMatchs[sub_idd]['match_sign_id'] and \
                haveMatchs[sub]['sign_id'] != haveMatchs[sub_idd]['sign_id'] :
                    singleFlag = 1
                    if haveMatchs[sub] not in duplicatedMatchs:
                        duplicatedMatchs.append(haveMatchs[sub])
                    if haveMatchs[sub_idd] not in duplicatedMatchs:
                        duplicatedMatchs.append(haveMatchs[sub_idd])
                    if haveMatchs[sub_idd]['match_sign_id'] + '_' + haveMatchs[sub_idd]['match_pic'] + '_' + haveMatchs[sub_idd]['sign_pic'] not in duplicatedList:
                        duplicatedList.append(haveMatchs[sub_idd]['match_sign_id'] + '_' + haveMatchs[sub_idd]['match_pic'] + '_' + haveMatchs[sub_idd]['sign_pic'])
        if singleFlag == 0:
            singleMatchs.append(haveMatchs[sub])

    duplicatedGroups = []
    for item in duplicatedList:
        duplicatedGroups.append({item:[]})

    for duplicated in duplicatedMatchs:
        for item in duplicatedGroups:
            if list(item.keys())[0] == duplicated['match_sign_id'] + '_' + duplicated['match_pic'] + '_' + duplicated['sign_pic']:
                item[duplicated['match_sign_id'] + '_' + duplicated['match_pic'] + '_' + duplicated['sign_pic']].append(duplicated)
	
	# Process filtering
	# 开始过滤
    keepDuplicateds = []
    for cc, duplicate in enumerate(duplicatedGroups):
        values_list = list(duplicate.values())
        for values in values_list:
            if postMethod == 'combine':
                image_sign = cv2.imread(picFolder + values[0]['sign_pic'] + '.jpg')
                image_sign_match = cv2.imread(picFolder + values[0]['match_pic'] + '.jpg')
            errors = []
            distances = []
            feature_errors = []
            hashErrors = []
            combine_errors = []
            for value in values:
                negBox = []
                for sign_info in preJson['signs']:
                    if sign_info['sign_id'] == value['sign_id']:
                        gtBox = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]
                    else:
                        negBox.append([sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])])

                    if sign_info['sign_id'] == value['match_sign_id']:
                        gtBox_match = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]
                distances.append(value['match_score'])

                if postMethod == 'combine':
                    error, feature_error, hashError = getSiftMatchScore(image_sign, image_sign_match, gtBox, negBox, gtBox_match, 3.0)
                    errors.append(error)
                    feature_errors.append(feature_error)
                    hashErrors.append(hashError)
                    combine_errors.append(error + value['match_score'] + hashError * 0.5)

            best_dis = min(distances)

            if postMethod == 'easy':
                for i, distance in enumerate(distances):
                    if distances[i] <= best_dis + confidenceRange:
                        keepDuplicateds.append(values[i])
            elif postMethod == 'combine':
                # best_error = min(errors)
                # best_fea = min(feature_errors)
                best_hash = min(hashErrors)
                best_combine = min(combine_errors)
                votes = [0] * len(errors)
                for i, distance in enumerate(distances):
                    if feature_errors[i] == 1.0:
                        if distances[i] <= best_dis + confidenceRange:
                            keepDuplicateds.append(values[i])
                            votes[i] = -1.0
                    else:
                        if distances[i] == best_dis:
                            votes[i] = votes[i] + 1
                        if hashErrors[i] == best_hash:
                            votes[i] = votes[i] + 1
                        if combine_errors[i] == best_combine:
                            votes[i] = votes[i] + 1

                for i, vote in enumerate(votes):
                    if vote <= 1:
                        pass
                    else:
                        keepDuplicateds.append(values[i])

    return singleMatchs, keepDuplicateds, noMatchs


def filterOneToMultiMatch(preJson, keepDuplicateds, postMethod, picFolder, confidenceRange):
    # Filter one to multi match
	# 针对一个匹配上同一张图多个目标的情况，进行过滤处理
    newsingleMatchs = []
    duplicatedMatchs = []
    duplicatedList = []
    haveMatchs = keepDuplicateds
	# Get one-to-multi pairs and collect them
	# 寻找一对多的匹配对
    for sub in range(len(haveMatchs)):
        singleFlag = 0
        for sub_idd in range(len(haveMatchs)):
            if haveMatchs[sub]['sign_pic'] == haveMatchs[sub_idd]['sign_pic'] and \
                            haveMatchs[sub]['match_pic'] == haveMatchs[sub_idd]['match_pic'] and \
                            haveMatchs[sub]['match_sign_id'] != haveMatchs[sub_idd]['match_sign_id'] and \
                            haveMatchs[sub]['sign_id'] == haveMatchs[sub_idd]['sign_id']:
                singleFlag = 1
                if haveMatchs[sub] not in duplicatedMatchs:
                    duplicatedMatchs.append(haveMatchs[sub])
                if haveMatchs[sub_idd] not in duplicatedMatchs:
                    duplicatedMatchs.append(haveMatchs[sub_idd])
                if haveMatchs[sub_idd]['sign_id'] + '_' + haveMatchs[sub_idd]['sign_pic'] + '_' + haveMatchs[sub_idd]['match_pic'] not in duplicatedList:
                    duplicatedList.append(haveMatchs[sub_idd]['sign_id'] + '_' + haveMatchs[sub_idd]['sign_pic'] + '_' + haveMatchs[sub_idd]['match_pic'])
        if singleFlag == 0:
            newsingleMatchs.append(haveMatchs[sub])

    duplicatedGroups = []
    for item in duplicatedList:
        duplicatedGroups.append({item: []})

    for duplicated in duplicatedMatchs:
        for item in duplicatedGroups:
            if list(item.keys())[0] == duplicated['sign_id'] + '_' + duplicated['sign_pic'] + '_' + duplicated['match_pic']:
                item[duplicated['sign_id'] + '_' + duplicated['sign_pic'] + '_' + duplicated['match_pic']].append(duplicated)
	
	# Process filtering
	# 开始过滤
    newkeepDuplicateds = []
    for cc, duplicate in enumerate(duplicatedGroups):
        values_list = list(duplicate.values())
        for values in values_list:
            if postMethod == 'combine':
                image_sign = cv2.imread(picFolder + values[0]['sign_pic'] + '.jpg')
                image_sign_match = cv2.imread(picFolder + values[0]['match_pic'] + '.jpg')
            errors = []
            distances = []
            combine_errors = []
            feature_errors = []
            hashErrors = []
            for value in values:
                negBox = []
                for sign_info in preJson['signs']:
                    if sign_info['sign_id'] == value['sign_id']:
                        gtBox = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]
                    else:
                        negBox.append([sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])])

                    if sign_info['sign_id'] == value['match_sign_id']:
                        gtBox_match = [sign_info['x'], sign_info['y'], (sign_info['x'] + sign_info['w']), (sign_info['y'] + sign_info['h'])]
                distances.append(value['match_score'])

                if postMethod == 'combine':
                    error, feature_error, hashError = getSiftMatchScore(image_sign, image_sign_match, gtBox, negBox, gtBox_match, 3.0)
                    errors.append(error)
                    feature_errors.append(feature_error)
                    combine_errors.append(error + value['match_score'] + hashError * 0.5)
                    hashErrors.append(hashError)

            best_dis = min(distances)
            if postMethod == 'easy':
                for i, distance in enumerate(distances):
                    if distances[i] <= best_dis + confidenceRange:
                        newkeepDuplicateds.append(values[i])
            elif postMethod == 'combine':
                # best_error = min(errors)
                # best_fea = min(feature_errors)
                best_dis = min(distances)
                best_hash = min(hashErrors)
                best_combine = min(combine_errors)
                votes = [0] * len(errors)
                for i, error in enumerate(errors):
                    if feature_errors[i] == 1.0:
                        if distances[i] <= best_dis + confidenceRange:
                            newkeepDuplicateds.append(values[i])
                            votes[i] = -1.0
                    else:
                        if distances[i] == best_dis:
                            votes[i] = votes[i] + 1
                        if hashErrors[i] == best_hash:
                            votes[i] = votes[i] + 1
                        if combine_errors[i] == best_combine:
                            votes[i] = votes[i] + 1

                for i, vote in enumerate(votes):
                    if vote <= 1:
                        pass
                    else:
                        newkeepDuplicateds.append(values[i])
    return newsingleMatchs, newkeepDuplicateds


def getLostMatchBack(preJson, newMatchs, picFolder, feature_signs, match_a):
	# Base on the similarities in one sequence, find same objects. And then make analysis to their pair information.
	# Get lost pair back
	# 通过单一序列内的相似度匹配，找到同一目标，然后针对可靠的匹配进行互补，将丢失的匹配对补回来
    have_match_ids = []
    no_match_ids = []
    for sign_info in preJson['signs']:
        flag = 0
        for match in newMatchs:
            if sign_info['sign_id'] == match['sign_id']:
                infos = [sign_info['sign_id'], sign_info['type'], sign_info['pic_id'],
                         [sign_info['x'], sign_info['y'], sign_info['x'] + sign_info['w'],
                          sign_info['y'] + sign_info['h']]]
                if infos not in have_match_ids:
                    have_match_ids.append(infos)
                flag = 1
        if flag == 0 and 'B' not in sign_info['sign_id']:
            infos = [sign_info['sign_id'], sign_info['type'], sign_info['pic_id'],
                     [sign_info['x'], sign_info['y'], sign_info['x'] + sign_info['w'], sign_info['y'] + sign_info['h']]]
            if infos not in no_match_ids:
                no_match_ids.append(infos)


    # make lost one-to-one match
    getHaveMatchBacks = []
    for pic_id in preJson['group'][0]['pic_list']:
        for class_id in classes:
            tmpSign = []
            tmpMatch = []

            class_count = 0
            for sign in preJson['signs']:
                if sign['pic_id'] == pic_id and sign['type'] == class_id:
                    class_count += 1
                    tmpSign = sign
            if class_count == 1:
                for pic_id_match in preJson['group'][1]['pic_list']:
                    match_class_count = 0
                    for sign_match in preJson['signs']:
                        if sign_match['pic_id'] == pic_id_match and sign_match['type'] == class_id:
                            match_class_count += 1
                            tmpMatch = sign_match
                    if match_class_count == 1:
                        have_flag = 0
                        for matches in newMatchs:
                            if tmpSign['sign_id'] == matches['sign_id'] and tmpMatch['sign_id'] == matches['match_sign_id']:
                                have_flag = 1
                        if have_flag == 0:
                            if tmpSign['type'] != '301':
                                getHaveMatchBacks.append(
                                    {'sign_type': tmpSign['type'], 'match_score': 0.15,
                                     'match_sign_id': tmpMatch['sign_id'],
                                     'sign_id': tmpSign['sign_id'], 'match_pic': tmpMatch['pic_id'],
                                     'match_type': tmpMatch['type'],
                                     'sign_pic': tmpSign['pic_id']})


    matches_301 = []
    matches_no_301 = []
    newMatchs = newMatchs + getHaveMatchBacks
    for match in newMatchs:
        if match['sign_type'] == '301':
            matches_301.append(match)
        else:
            matches_no_301.append(match)

    found_names = []
    found_namesB = []
    for match301 in matches_301:
        if match301['sign_id'] not in found_names and 'B' not in match301['sign_id']:
            found_names.append(match301['sign_id'])
        if match301['match_sign_id'] not in found_namesB and 'B' in match301['match_sign_id']:
            found_namesB.append(match301['match_sign_id'])

    found_matches = []
    for sign_id in found_names:
        single_match = []
        for match in matches_301:
            if match['sign_id'] == sign_id:
                single_match.append(match['match_sign_id'])
        found_matches.append({sign_id:single_match})


    #=========================== Seq A ===================================
    if len(found_names) >= 1:
        features = sign2feature(found_names, feature_signs)
        score_matrix = cosine_distancesmall(features)
        # print(score_matrix)

    best_pairs = []
    for index, nameA in enumerate(found_names):
        bp = []
        for idx in range(score_matrix[index].shape[0]):
            if score_matrix[index][idx] <= 0.03:
                bp.append(found_names[idx])
        best_pairs.append({found_names[index]:bp})

    sameObjects = []
    tmpObjects = []
    for index, bestP in enumerate(best_pairs):
        pair = []
        pair.append(list(bestP.keys())[0])
        for item in list(bestP.values())[0]:
            pair.append(item)

        if sameObjects == []:
            tmpObjects.append(pair)
        else:
            flag = 0
            for pid, ppa in enumerate(sameObjects):
                intersect = list(set(pair).intersection(set(ppa)))
                ratio = len(intersect) / float(max(len(pair), len(ppa)))
                if ratio >= 0.5:
                    flag = 1
                    tmpObjects[pid] = list(set(ppa).union(set(pair)))
            if flag == 0:
                tmpObjects.append(pair)
        sameObjects = tmpObjects.copy()

    new_foundMatchs = found_matches.copy()
    for sameObj in sameObjects:
        if len(sameObj) >= 4:
            print('--------------------------------------------------------')
            goodPairs = []
            for indexObj in range(len(sameObj)):
                for pairDict in new_foundMatchs:
                    if list(pairDict.keys())[0] == sameObj[indexObj]:
                        goodPairs = goodPairs + list(pairDict.values())[0]

            totalObjs = list(set(goodPairs))
            totalCounts = [0] * len(totalObjs)
            for iiiii, obj in enumerate(totalObjs):
                for obj1 in goodPairs:
                    if obj == obj1:
                        totalCounts[iiiii] += 1
            new_goodPairs = []
            for iiiii in range(len(totalCounts)):
                if totalCounts[iiiii] >= (len(sameObj) - 3):
                    new_goodPairs.append(totalObjs[iiiii])

            for indexObj in range(len(sameObj)):
                for iii, pairDict in enumerate(new_foundMatchs):
                    if list(pairDict.keys())[0] == sameObj[indexObj]:
                        new_foundMatchs[iii] = {found_names[iii]: new_goodPairs}
    found_matches = new_foundMatchs.copy()
    #======================================================================


    # =========================== Seq B ===================================
    if len(found_namesB) >= 1:
        featuresB = sign2feature(found_namesB, feature_signs)
        score_matrixB = cosine_distancesmall(featuresB)

        for idxPair, pairDict in enumerate(found_matches):
            matchPairs = list(pairDict.values())[0]
            postPairs = matchPairs.copy()
            for index, nameB in enumerate(found_namesB):
                for matchPair in matchPairs:
                    if matchPair == nameB:
                        scoreListB = score_matrixB[index]
                        for idxs, score in enumerate(scoreListB):
                            if score < 0.02:
                                postPairs.append(found_namesB[idxs])

            postPairs = set(postPairs)
            if len(matchPairs) != len(postPairs):
                found_matches[idxPair] = {found_names[idxPair]: postPairs}
                # print('--------------------------------------------------------')
    # ======================================================================


    getNewMatches = []
    for match in found_matches:
        sign_id = list(match.keys())[0]
        matches_ids = list(match.values())[0]
        pic_id = ''
        sign_type = ''
        for sign in preJson['signs']:
            if sign['sign_id'] == sign_id:
                pic_id = sign['pic_id']
                sign_type = sign['type']

        for match_sign_id in matches_ids:
            for sign in preJson['signs']:
                if sign['sign_id'] == match_sign_id:
                    getNewMatches.append({'sign_type': sign_type, 'match_score': 0.15,
                                     'match_sign_id': match_sign_id,
                                     'sign_id': sign_id, 'match_pic': sign['pic_id'],
                                     'match_type': sign_type,
                                     'sign_pic': pic_id})

    newMatchs = getNewMatches + matches_no_301

    return newMatchs


def getNonMatchBack(preJson, newMatchs):
	# Get all rest lost objects back, and write NONE Match.
	# 将所有没有匹配上的目标找回，并写入空匹配
    getHaveMatchBacks = []
    for sign_info in preJson['signs']:
        flag = 0
        for match in newMatchs:
            if sign_info['sign_id'] == match['sign_id'] or sign_info['sign_id'] == match['match_sign_id']:
                flag = 1
        if flag == 0:
            getHaveMatchBacks.append({'sign_id': sign_info['sign_id'], 'match_sign_id': ' ', 'match_score':1.0, 'sign_pic':'', 'match_pic': '', 'sign_type':'', 'match_type':''})
            # if 'B' in sign_info['sign_id']:
            #     print({'sign_id': sign_info['sign_id'], 'match_sign_id': ' ', 'match_score':1.0, 'sign_pic':'', 'match_pic': '', 'sign_type':'', 'match_type':''})
    postMatchs = deleteDuplicatedElementFromList(newMatchs + getHaveMatchBacks)
    return postMatchs