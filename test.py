import torch as t
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
from tqdm import tqdm



def calcRes(topLocs, tstLocs, batIds):
    assert topLocs.shape[0] == len(batIds)
    allRecall = allNdcg = 0
    for i in range(len(batIds)):
        temTopLocs = list(topLocs[i])
        temTstLocs = tstLocs[batIds[i]]
        tstNum = len(temTstLocs)
        maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
        recall = dcg = 0
        for val in temTstLocs:
            if val in temTopLocs:
                recall += 1
                dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
        recall = recall / tstNum
        ndcg = dcg / maxDcg
        allRecall += recall
        allNdcg += ndcg
    return allRecall, allNdcg

if __name__ == '__main__':
    log('Start test')
    print(args)
    handler = DataHandler()
    handler.LoadData()

    model = Model(handler.trnMat, handler.img_feats, handler.txt_feats).cuda()
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
    path = './save/'+ args.data + '.pth'
    checkpoint = t.load(path)
    print(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    tstLoader = handler.tstLoader
    epLoss, epRecall, epNdcg= [0] * 3
    i = 0
    num = tstLoader.dataset.__len__()
    steps = num // args.tstBat
    with t.no_grad():
        for usr, trnMask in tqdm(tstLoader, desc='Calculating testset ...', unit='batch'):
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds = model(handler.torchBiAdj, handler.torchBiAdj)
            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = calcRes(topLocs.cpu().numpy(), handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg

    ret = dict()
    ret['Recall'] = epRecall / num
    ret['NDCG'] = epNdcg / num

    print(ret)