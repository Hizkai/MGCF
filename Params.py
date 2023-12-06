import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--reg', default=4e-4, type=float, help='weight decay regularizer')
	parser.add_argument('--centralNodes', default=200, type=int, help='number of masked central nodes')
	parser.add_argument('--scheduler',default=10, type=int,help='opt scheduler')


	parser.add_argument('--batch', default=1024, type=int, help='batch size')
	parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--head', default=4, type=int, help='number of heads in attention')
	parser.add_argument('--gcn_layer', default=5, type=int, help='number of gcn layers')
	parser.add_argument('--gt_layer', default=1, type=int, help='number of graph transformer layers')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='baby', type=str, help='name of dataset')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--maskDepth', default=2, type=int, help='depth to mask')
	parser.add_argument('--ssl_reg', default=0.3, type=float, help='contrastive regularizer')
	parser.add_argument('--m_rate',default=0.55, type=float,help='modality feature rate')
	parser.add_argument('--fixSteps', default=10, type=int, help='steps to train on the same sampled graph')
	parser.add_argument('--keepRate', default=0.2, type=float, help='ratio of nodes to keep')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument('--eps', default=0.2, type=float, help='scaled weight as reward')	
	parser.add_argument('--early_stopping_patience', default=7, type=int, help='epochs to early stop')
	
	return parser.parse_args()

args = ParseArgs()