import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
from Utils.Utils import contrast,EmbLoss

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

		self.reg_loss = EmbLoss()

	

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		self.model_info()

		stloc = 0
		bestRes = None
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses, usrEmbeds, itmEmbeds = self.testEpoch()
				log(self.makePrint('Test', ep, reses, tstFlag))
				if bestRes is None or reses['Recall'] > bestRes['Recall']:
					bestRes = reses
					stopping_step = 0
				elif stopping_step < args.early_stopping_patience:
					stopping_step += 1
					print('#####Early stopping steps: %d #####' % stopping_step)
				else:
					print('#####Early stop! #####')
					break
			print()

		reses,_,_ = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('Best Result', args.epoch, bestRes, True))

	def prepareModel(self):
		self.model = Model(self.handler.trnMat, self.handler.img_feats, self.handler.txt_feats).cuda()
		self.masker = RandomMaskSubgraphs()
		self.sampler = LocalGraph()
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
		self.scheduler_D = self.set_lr_scheduler()
	
	def model_info(self):
		print("--------trainable Parameters---------")
		print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
		for name,parameters in self.model.named_parameters():
			print(name,':',parameters.size())

	def set_lr_scheduler(self):
		fac = lambda epoch: 0.96 ** (epoch / args.scheduler)
		scheduler_D = t.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=fac)
		return scheduler_D  
	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epPreLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			if i % args.fixSteps == 0:
				embeds = self.model.getMoEmbeds()
				sampScores, seeds = self.sampler(self.handler.allOneAdj, embeds)
				encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds)
			ancs, poss, _ = tem

			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			usrEmbeds, itmEmbeds = self.model(encoderAdj, decoderAdj)

			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]

			bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
			regLoss = calcRegLoss(self.model) * args.reg
			contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(ancs, usrEmbeds, itmEmbeds)

			loss = bprLoss + regLoss + contrastLoss 

			if i % args.fixSteps == 0:
				localGlobalLoss = -sampScores.mean()
				loss += localGlobalLoss
			epLoss += loss.item()
			epPreLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			log('Step %d/%d: loss = %.1f, reg = %.1f, cl = %.1f   ' % (i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)

		self.scheduler_D.step()
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['preLoss'] = epPreLoss / steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epLoss, epRecall, epNdcg = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		with t.no_grad():
			for usr, trnMask in tstLoader:
				i += 1
				usr = usr.long().cuda()
				trnMask = trnMask.cuda()
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)

				allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
	
				_, topLocs = t.topk(allPreds, args.topk)
				recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
				epRecall += recall
				epNdcg += ndcg

			ret = dict()
			ret['Recall'] = epRecall / num
			ret['NDCG'] = epNdcg / num

		return ret, usrEmbeds, itmEmbeds

	def calcRes(self, topLocs, tstLocs, batIds):
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
	logger.saveDefault = True
	log('Start')
	print(args)
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()