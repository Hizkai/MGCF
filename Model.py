from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import scipy.sparse as sp

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, trnMat, image_feats, text_feats):
		super(Model, self).__init__()

		self.iu_graph = trnMat.T
		self.ui_graph = self.matrix_to_tensor(self.csr_norm(trnMat, mean_flag=True))
		self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))


		self.uEmbeds = nn.Embedding(args.user, args.latdim)
		self.iEmbeds = nn.Embedding(args.item, args.latdim)
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer)])
		self.gtLayers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])


		self.imgFeats = t.tensor(image_feats).float().cuda()
		self.img_trans = nn.Linear(self.imgFeats.shape[1], args.latdim)
		self.txtFeats = t.tensor(text_feats).float().cuda()
		self.txt_trans = nn.Linear(self.txtFeats.shape[1], args.latdim)

		nn.init.xavier_uniform_(self.uEmbeds.weight)
		nn.init.xavier_uniform_(self.iEmbeds.weight)
		
		nn.init.xavier_uniform_(self.img_trans.weight)
		nn.init.xavier_uniform_(self.txt_trans.weight)

	def mm(self, x, y):
		return t.mm(x, y)
		
	def csr_norm(self, csr_mat, mean_flag=False):
		rowsum = np.array(csr_mat.sum(1))  
		rowsum = np.power(rowsum+1e-8, -0.5).flatten()
		rowsum[np.isinf(rowsum)] = 0.
		rowsum_diag = sp.diags(rowsum) 

		colsum = np.array(csr_mat.sum(0))  
		colsum = np.power(colsum+1e-8, -0.5).flatten()
		colsum[np.isinf(colsum)] = 0.
		colsum_diag = sp.diags(colsum) 

		if mean_flag == False:
			return rowsum_diag*csr_mat*colsum_diag
		else:
			return rowsum_diag*csr_mat
		
	def matrix_to_tensor(self, cur_matrix):
		if type(cur_matrix) != sp.coo_matrix:
			cur_matrix = cur_matrix.tocoo()  
		indices = t.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
		values = t.from_numpy(cur_matrix.data)  
		shape = t.Size(cur_matrix.shape)
		return t.sparse.FloatTensor(indices, values, shape).to(t.float32).cuda()
		
		
	def getMoEmbeds(self):
		imgFeats = self.img_trans(self.imgFeats)
		txtFeats = self.txt_trans(self.txtFeats)

		img_u_feats = self.mm(self.ui_graph, imgFeats)
		img_i_feats = self.mm(self.iu_graph, img_u_feats)

		txt_u_feats = self.mm(self.ui_graph, txtFeats)
		txt_i_feats = self.mm(self.iu_graph, txt_u_feats)

		uFeats = t.concat([img_u_feats, txt_u_feats], dim=-1)
		iFeats = t.concat([img_i_feats, txt_i_feats], dim=-1)

		return t.concat([uFeats,iFeats])

	def forward(self, encoderAdj, decoderAdj=None):

		imgFeats = self.img_trans(self.imgFeats)
		txtFeats = self.txt_trans(self.txtFeats)

		img_u_feats = self.mm(self.ui_graph, imgFeats)
		img_i_feats = self.mm(self.iu_graph, img_u_feats)

		txt_u_feats = self.mm(self.ui_graph, txtFeats)
		txt_i_feats = self.mm(self.iu_graph, txt_u_feats)


		cl_embeds = t.concat([self.uEmbeds.weight, self.iEmbeds.weight])
		cl_embedsLst = [cl_embeds]
		for i, gcn in enumerate(self.gcnLayers):
			cl_embeds = gcn(encoderAdj, cl_embedsLst[-1])
			cl_embedsLst.append(cl_embeds)
		if decoderAdj is not None:
			for gt in self.gtLayers:
				cl_embeds = gt(decoderAdj, cl_embedsLst[-1])
				cl_embedsLst.append(cl_embeds)
		cl_embeds = sum(cl_embedsLst)

		u_embeds = cl_embeds[:args.user] + args.m_rate * (F.normalize(img_u_feats,p=2,dim=1) + F.normalize(txt_u_feats,p=2,dim=1))
		i_embeds = cl_embeds[args.user:] + args.m_rate * (F.normalize(img_i_feats,p=2,dim=1) + F.normalize(txt_i_feats,p=2,dim=1))

		return u_embeds, i_embeds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class GTLayer(nn.Module):
	def __init__(self):
		super(GTLayer, self).__init__()
		self.qTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.kTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.vTrans = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
	
	def forward(self, adj, embeds):
		indices = adj._indices()
		rows, cols = indices[0, :], indices[1, :]
		rowEmbeds = embeds[rows]
		colEmbeds = embeds[cols]

		qEmbeds = (rowEmbeds @ self.qTrans).view([-1, args.head, args.latdim // args.head])
		kEmbeds = (colEmbeds @ self.kTrans).view([-1, args.head, args.latdim // args.head])
		vEmbeds = (colEmbeds @ self.vTrans).view([-1, args.head, args.latdim // args.head])
		att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
		att = t.clamp(att, -10.0, 10.0)
		expAtt = t.exp(att)
		tem = t.zeros([adj.shape[0], args.head]).cuda()
		attNorm = (tem.index_add_(0, rows, expAtt))[rows]
		att = expAtt / (attNorm + 1e-8)
		
		resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, args.latdim])
		tem = t.zeros([adj.shape[0], args.latdim]).cuda()
		resEmbeds = tem.index_add_(0, rows, resEmbeds) 
		return resEmbeds

class LocalGraph(nn.Module):
	def __init__(self):
		super(LocalGraph, self).__init__()
	
	def makeNoise(self, scores):
		noise = t.rand(scores.shape).cuda()
		noise = -t.log(-t.log(noise))
		return t.log(scores) + noise
	
	def forward(self, allOneAdj, embeds):
		order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
		fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
		fstNum = order
		scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
		scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
		subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
		subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
		embeds = F.normalize(embeds, p=2)
		scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
		scores = self.makeNoise(scores)
		_, seeds = t.topk(scores, args.centralNodes)
		return scores, seeds

class RandomMaskSubgraphs(nn.Module):
	def __init__(self):
		super(RandomMaskSubgraphs, self).__init__()
		self.flag = False
	
	def normalizeAdj(self, adj):
		degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
		newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
		rowNorm, colNorm = degree[newRows], degree[newCols]
		newVals = adj._values() * rowNorm * colNorm
		return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

	def forward(self, adj, seeds):
		rows = adj._indices()[0, :]
		cols = adj._indices()[1, :]

		maskNodes = [seeds]

		for i in range(args.maskDepth):
			curSeeds = seeds if i == 0 else nxtSeeds
			nxtSeeds = list()
			for seed in curSeeds:
				rowIdct = (rows == seed)
				colIdct = (cols == seed)
				idct = t.logical_or(rowIdct, colIdct)

				if i != args.maskDepth - 1:
					mskRows = rows[idct]
					mskCols = cols[idct]
					nxtSeeds.append(mskRows)
					nxtSeeds.append(mskCols)

				rows = rows[t.logical_not(idct)]
				cols = cols[t.logical_not(idct)]
			if len(nxtSeeds) > 0:
				nxtSeeds = t.unique(t.concat(nxtSeeds))
				maskNodes.append(nxtSeeds)
		sampNum = int((args.user + args.item) * args.keepRate)
		sampedNodes = t.randint(args.user + args.item, size=[sampNum]).cuda()
		if self.flag == False:
			l1 = adj._values().shape[0]
			l2 = rows.shape[0]
			print('-----')
			print('MASKED INTERACTIONS:', '%d' % (l1-l2))
			tem = t.unique(t.concat(maskNodes))
		maskNodes.append(sampedNodes)
		maskNodes = t.unique(t.concat(maskNodes))
		if self.flag == False:
			self.flag = True
			print('-----')

		
		encoderAdj = self.normalizeAdj(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

		temNum = maskNodes.shape[0]
		temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
		temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]

		newRows = t.concat([temRows, temCols, t.arange(args.user+args.item).cuda(), rows])
		newCols = t.concat([temCols, temRows, t.arange(args.user+args.item).cuda(), cols])

		hashVal = newRows * (args.user + args.item) + newCols
		hashVal = t.unique(hashVal)
		newCols = hashVal % (args.user + args.item)
		newRows = ((hashVal - newCols) / (args.user + args.item)).long()


		decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(), adj.shape)
		return encoderAdj, decoderAdj
