import os
import glob
import cv2
import argparse
import numpy as np

# SBD by reference
def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice

def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)


# SBD by own made
def BD(img_a, img_b):

    ra, ca = img_a.shape
    rb, cb = img_b.shape

    a_list, b_list = {}, {}

    for r in range(ra):
        for c in range(ca):
            if img_a[r][c] not in a_list:
                a_list[img_a[r][c]] = [(r, c)]
            else:
                temp = a_list[img_a[r][c]]
                temp.append((r, c))
                a_list[img_a[r][c]] = temp

    for r in range(rb):
        for c in range(cb):
            if img_b[r][c] not in b_list:
                b_list[img_a[r][c]] = [(r, c)]
            else:
                temp = b_list[img_b[r][c]]
                temp.append((r, c))
                b_list[img_b[r][c]] = temp

    M = len(a_list)
    N = len(b_list)

    max_list = list()
    for i in a_list:
        result = list()
        La = a_list[i]
        size_a = len(La)
        for j in b_list:
            Lb = b_list[j]
            size_b = len(Lb)

            cover = 0
            # check if cover
            for coordinate in Lb:
                if coordinate in La:
                    cover+=1

            result.append(2*cover/(size_a+size_b))

        max_list.append(max(result))

    bd = sum(max_list)
    bd = bd/M
    return bd


def SBD(img_a, img_b):
    return min(BD(img_a, img_b), BD(img_b, img_a))


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--dataset', type=str, required=True, help='Input predict dataset')
    parse.add_argument('-l', '--label_dataset', type=str, required=False, help='Label dataset path')
    args = parse.parse_args() 

    file_path = glob.glob(os.path.join(args.dataset, '*_ws.png'))

    sbd_record = list()

    for file_name in file_path:
        image_name = file_name.replace('_ws.png', '')

        predict = cv2.imread(image_name+'_ws.png', 0)
        img_true = cv2.imread(image_name+'_label.png', 0)

        score = calc_sbd(img_true, predict)
        sbd_record.append(score)

        print(file_name, '%.3f'%score)

    print('overall mean sbd:', np.mean(sbd_record))