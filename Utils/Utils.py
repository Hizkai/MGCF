import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def contrast(nodes, allEmbeds, allEmbeds2=None):
	if allEmbeds2 is not None:
		pckEmbeds = allEmbeds[nodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
	else:
		uniqNodes = t.unique(nodes)
		pckEmbeds = allEmbeds[uniqNodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
	return scores


def cosine(allEmbeds, allEmbeds2):
	scores = 1 - cosine_similarity(allEmbeds, allEmbeds2, dim=-1).mean()
	return scores

class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = t.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += t.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss