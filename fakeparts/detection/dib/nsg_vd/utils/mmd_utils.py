import numpy as np
import torch
import os
import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
# os.chdir("/U_20240905_ZSH_SMIL/lzh/DeT/")
from sklearn.metrics.pairwise import pairwise_distances
try:
	from loguru import logger
except ImportError:
		import logging
		logger = logging.getLogger(__name__)

is_cuda = True

def guassian_kernel(source, target, kernel_mul, kernel_num=5, fix_sigma=None):
	"""计算Gram核矩阵
	source: sample_size_1 * feature_size 的数据
	target: sample_size_2 * feature_size 的数据
	kernel_mul: 这个概念不太清楚,感觉也是为了计算每个核的bandwith
	kernel_num: 表示的是多核的数量
	fix_sigma: 表示是否使用固定的标准差
		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[   K_ss K_st
							K_ts K_tt ]
	"""
	n_samples = int(source.size()[0])+int(target.size()[0])
	total = torch.cat([source, target], dim=0) # 合并在一起

	total0 = total.unsqueeze(0).expand(int(total.size(0)), \
									int(total.size(0)), \
									int(total.size(1)))
	total1 = total.unsqueeze(1).expand(int(total.size(0)), \
									int(total.size(0)), \
									int(total.size(1)))
	value_factor = 15
	L2_distance = (((total0-total1)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|
	# L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
	scale_factor = 0
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# bandwidth_list = [torch.clamp(bandwidth * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]

	# 高斯核的公式，exp(-|x-y|/bandwith)
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list]

	return sum(kernel_val)/kernel_num # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	n = int(source.size()[0])
	m = int(target.size()[0])

	kernels = guassian_kernel(source, target,
							kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
	XX = kernels[:n, :n] 
	YY = kernels[n:, n:]
	XY = kernels[:n, n:]
	YX = kernels[n:, :n]

	XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
	XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

	YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
	YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
		
	loss = (XX + XY).sum() + (YX + YY).sum()
	# loss = XY.sum()
	return loss

def L2_distance_get(x, y, value_factor = 1):
	"""compute the paired distance between x and y."""
	x_norm = ((x/value_factor) ** 2).sum(1).view(-1, 1)
	y_norm = ((y/value_factor) ** 2).sum(1).view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x/value_factor, torch.transpose(y/value_factor, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def mmd_guassian_bigtensor(L2_distance_xx, L2_distance_yx, value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):


	x_num = L2_distance_xx.shape[0]
	y_num = L2_distance_yx.shape[0]
	assert y_num==1 and L2_distance_yx.shape[1]==x_num
	
	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = (torch.sum(L2_distance_xx/x_num) + torch.sum(L2_distance_yx/x_num))/(x_num-1 + L2_distance_yx.shape[1])
		# bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
		assert not torch.isinf(bandwidth).any()
	# bandwidth /= kernel_mul ** (kernel_num // 2)
	bandwidth = bandwidth / kernel_mul ** (kernel_num // 2)
	# print("bandwidth:",bandwidth)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# 高斯核的公式，exp(-|x-y|/bandwith)
	XX = sum([torch.exp(-L2_distance_xx / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list])/kernel_num
	YX = sum( [torch.exp(-L2_distance_yx / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list])/kernel_num

	L2_distance = (XX/x_num).sum()/x_num - 2*YX.sum()/x_num
	return L2_distance

def mmd_guassian_bigtensor2(L2_distance_xx, L2_distance_yx, value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):


	x_num = L2_distance_xx.shape[0]
	y_num = L2_distance_yx.shape[0]
	assert y_num==1 and L2_distance_yx.shape[1]==x_num
	
	bandwidth = fix_sigma
	bandwidth /= kernel_mul ** (kernel_num // 2)
	# print("bandwidth:",bandwidth)
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# 高斯核的公式，exp(-|x-y|/bandwith)
	YX = sum( [torch.exp(-L2_distance_yx / bandwidth_temp) for \
				bandwidth_temp in bandwidth_list])/kernel_num

	L2_distance = - 2*YX.sum()/x_num
	return L2_distance

def mmd_guassian_kernel_batch(ref_data, test_data, kernel_num=5, kernel_mul=2.0, fix_sigma=None, clean_flag=False):

	num_ref = ref_data.shape[0]
	# if clean_flag:
	# 	num_ref = ref_data.shape[0]-1
	num_test = test_data.shape[0]
	# ref_data_exp = ref_data.unsqueeze(0)
	# test_data_exp = test_data.unsqueeze(1)
	value_factor = 1
	# L2_distance = (((ref_data_exp-test_data_exp)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|
	L2_distance = L2_distance_get(test_data, ref_data, value_factor)

	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth = torch.sum(L2_distance.data / ((num_ref)/(value_factor)**2) /(value_factor)**2,dim=-1)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
	assert bandwidth.shape[0]==num_test
	bandwidth_alllist = []
	scale_factor = 1
	for id_test in range(num_test):
		# bandwidth_list = [torch.clamp(bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]
		bandwidth_list = [bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)) for i in range(kernel_num)]
		bandwidth_alllist.append(bandwidth_list)
	bandwidth_alltensor = torch.tensor(bandwidth_alllist,device=test_data.device)
	assert bandwidth_alltensor.shape[1]==kernel_num
	L2_distance_all = 0
	for k in range(kernel_num):
		L2_distance_all += (torch.exp(-L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)))
		# L2_distance_all += -L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)
	assert L2_distance_all.shape[0]==num_test

	# return L2_distance_all.sum(dim=-1)/(-kernel_num*num_ref)
	return L2_distance_all.sum(dim=-1)/(-num_ref)

	# 高斯核的公式，exp(-|x-y|/bandwith)
	# kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
	# 			  bandwidth_temp in bandwidth_list]

	# return sum(kernel_val)/kernel_num # 将多个核合并在一起
	
def mmd_guassian_kernel_single(L2_distance=0.0,value_factor = 15, kernel_num=5, kernel_mul=2.0,  fix_sigma=None, clean_flag=False):

	num_ref = L2_distance.shape[1]
	num_test = L2_distance.shape[0]
	# ref_data_exp = ref_data.unsqueeze(0)
	# test_data_exp = test_data.unsqueeze(1)
	
	# L2_distance = (((ref_data_exp-test_data_exp)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|

	assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'

	# 计算多核中每个核的bandwidth
	if fix_sigma is not None:
		bandwidth0 = fix_sigma.expand(int(num_test))
	else:
		# bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		bandwidth0 = torch.sum(L2_distance.data / ((num_ref)/(value_factor)**2) /(value_factor)**2,dim=-1)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth0/ kernel_mul ** (kernel_num // 2)
	assert bandwidth.shape[0]==num_test
	bandwidth_alllist = []
	scale_factor = 1
	for id_test in range(num_test):
		# bandwidth_list = [torch.clamp(bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]
		bandwidth_list = [bandwidth[id_test] * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)) for i in range(kernel_num)]
		bandwidth_alllist.append(bandwidth_list)
	bandwidth_alltensor = torch.tensor(bandwidth_alllist,device=L2_distance.device)
	assert bandwidth_alltensor.shape[1]==kernel_num
	L2_distance_all = 0
	for k in range(kernel_num):
		L2_distance_all += (torch.exp(-L2_distance / bandwidth_alltensor[:,k].unsqueeze(1)))
	assert L2_distance_all.shape[0]==num_test

	# return L2_distance_all.sum(dim=-1)/(-kernel_num*num_ref)
	return L2_distance_all.sum(dim=-1)/(-num_ref)

def get_item(x, is_cuda):
	"""get the numpy value from a torch tensor."""
	if is_cuda:
		x = x.cpu().detach().numpy()
	else:
		x = x.detach().numpy()
	return x

def MatConvert(x, device, dtype):
	"""convert the numpy to a torch tensor."""
	x = torch.from_numpy(x).to(device, dtype)
	return x

def Pdist2(x, y):
	"""compute the paired distance between x and y."""
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y = x
		y_norm = x_norm.view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def MMD_batch(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True):
	X = Fea_org[0:len_s, :]
	Y = Fea_org[len_s:, :]
	X_fea = Fea[0:len_s, :]
	Y_fea = Fea[len_s:, :]
	dis_vector = torch.zeros((Y.shape[0]),device=Fea.device)
	for i in range(Y.shape[0]):
		dis_vector[i] = MMDu(torch.cat((X_fea,Y_fea[[i]]),dim=0), len_s, torch.cat((X,Y[[i]]),dim=0).view(len_s+1,-1), sigma, sigma0, epsilon, is_unbiased=False)[0]
	return dis_vector

def MMD_batch2(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True, coeff_xy=2):
	X = Fea[0:len_s, :] # 400,300
	Y = Fea[len_s:, :] # 2000,300
	if is_smooth:
		X_org = Fea_org[0:len_s, :] # 400,76800
		Y_org = Fea_org[len_s:, :]  # 2000,76800
	L = 1 # generalized Gaussian (if L>1)

	nx = X.shape[0] # 400
	ny = Y.shape[0] # 2000
	Dxx = Pdist2(X, X) # 400,400
	Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
	# Dyy = Pdist2(Y, Y) # 2000, 1  0
	Dxy = Pdist2(X, Y).transpose(0,1) # 400，2000 => 2000, 400
	if is_smooth:
		Dxx_org = Pdist2(X_org, X_org)
		Dyy_org = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
		# Dyy_org = Pdist2(Y_org, Y_org) # 1，1  0
		Dxy_org = Pdist2(X_org, Y_org).transpose(0,1)  # 400，2000 => 2000, 400
	# K_Ix = torch.eye(nx).cuda() # 创建单位矩阵
	# K_Iy = torch.eye(ny).cuda() 

	if is_smooth:
		Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma) # 400,400
		Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma) # 2000,1     值 1
		Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)# 400,2000
	else:
		Kx = torch.exp(-Dxx / sigma0)
		Ky = torch.exp(-Dyy / sigma0)
		Kxy = torch.exp(-Dxy / sigma0)

	nx = Kx.shape[0] # 400
	# ny = 1 # 2000
	is_unbiased = False
	if 1:
		xx = torch.div((torch.sum(Kx)), (nx * nx)) # 400,400 => 1
		yy = Ky.reshape(-1) # 2000
		# yy = torch.div((torch.sum(Ky)), (ny * ny)) # 2000,1 => 2000 1  [1]
		# one-sample U-statistic.

		xy = torch.div(torch.sum(Kxy, dim = 1), (nx )) # 2000
		
		mmd2 = xx - 2 * xy + yy
	return mmd2

def compute_mmd(X, Y, sigma):
		"""mmd"""
		if sigma == 0:
			sigma = 1e-6 
		
		K_XX = np.exp(-pairwise_distances(X, metric='sqeuclidean') / (2 * sigma**2))
		K_YY = np.exp(-pairwise_distances(Y, metric='sqeuclidean') / (2 * sigma**2))
		K_XY = np.exp(-pairwise_distances(X, Y, metric='sqeuclidean') / (2 * sigma**2))
		n, m = len(X), len(Y)
		mean_XX = (np.sum(K_XX) - np.sum(np.diag(K_XX))) / (n * (n - 1)) if n > 1 else 0
		mean_YY = (np.sum(K_YY) - np.sum(np.diag(K_YY))) / (m * (m - 1)) if m > 1 else 0
		mean_XY = np.mean(K_XY)
		
		mmd_squared = mean_XX + mean_YY - 2 * mean_XY
		return np.sqrt(max(mmd_squared, 0)) 

def L2_distance_get(x, y, value_factor = 1):
	"""compute the paired distance between x and y."""
	x_norm = ((x/value_factor) ** 2).sum(1).view(-1, 1)
	y_norm = ((y/value_factor) ** 2).sum(1).view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x/value_factor, torch.transpose(y/value_factor, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def L2_distance(x, y, value_factor = 1):
	"""compute the paired distance between x and y."""
	xx = L2_distance_get(x/value_factor, x/value_factor)
	yy = L2_distance_get(y/value_factor, y/value_factor)
	xy = L2_distance_get(x/value_factor, y/value_factor)
	yx = L2_distance_get(y/value_factor, x/value_factor)
	xx_xy = torch.cat([xx, xy], dim=1)
	yx_yy = torch.cat([yx, yy], dim=1)
	Pdist = torch.cat([xx_xy, yx_yy], dim=0)
	Pdist[Pdist<0]=0
	return Pdist
	
class MMDLoss(nn.Module):
	def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
		super(MMDLoss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		self.kernel_type = kernel_type

	def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma, scale_factor = 1):
		n_samples = int(source.size()[0]) + int(target.size()[0])
		L2_distance_2 = L2_distance(source, target, scale_factor)
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance_2.data/scale_factor) / (n_samples**2-n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i)
						for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance_2 / bandwidth_temp)
					for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def linear_mmd2(self, f_of_X, f_of_Y):
		loss = 0.0
		delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
		loss = delta.dot(delta.T)
		return loss

	def forward(self, source, target, scale_factor=1):
		if self.kernel_type == 'linear':
			return self.linear_mmd2(source, target)
		elif self.kernel_type == 'rbf':
			batch_size = int(source.size()[0])
			kernels = self.guassian_kernel(
				source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma, scale_factor=scale_factor)
			XX = torch.mean(kernels[:batch_size, :batch_size])
			YY = torch.mean(kernels[batch_size:, batch_size:])
			XY = torch.mean(kernels[:batch_size, batch_size:])
			YX = torch.mean(kernels[batch_size:, :batch_size])
			loss = torch.mean(XX + YY - XY - YX)
			return loss
		
def plot_mi(clean, adv, plot=False):
	mi_nat = clean.numpy()
	label_clean = 'Clean'
	mi_svhn = adv.numpy()
	label_adv = 'Adv'

	mi_nat = mi_nat[~np.isnan(mi_nat)]
	mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

	# if False:
	# Draw the density plot
	if plot:
		import seaborn as sns

		fig = plt.figure()
		sns.distplot(mi_nat, hist = True, kde = True,
					kde_kws = {'shade': True, 'linewidth': 1},
					label = label_clean)
		sns.distplot(mi_svhn, hist = True, kde = True,
					kde_kws = {'shade': True, 'linewidth': 1},
					label = label_adv)
		fig.legend()

	x = np.concatenate((mi_nat, mi_svhn), 0)
	y = np.zeros(x.shape[0])
	y[mi_nat.shape[0]:] = 1

	ap = metrics.roc_auc_score(y, x)
	fpr, tpr, thresholds = metrics.roc_curve(y, x)
	accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
	thre = {tpr[np.argwhere(fpr <= th).max()]: thresholds[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

	# print("; ".join(["TPR: {:.4f} @ thresholds={:.4f}".format(k, v) for k, v in thre.items()]))

	return ap, "auroc: {:.4f}; TPR: {:.4f} @ FPR={:.4f}; ...; TPR: {:.4f} @ FPR={:.4f}; {}-{}"

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, is_unbiased = True, coeff_xy=2, is_yy_zero=True, is_xx_zero=False):
	"""compute value of MMD and std of MMD using kernel matrix."""
	if not is_yy_zero:
		coeff_yy = 1
	else:
		coeff_yy = 0
	if not is_xx_zero:
		coeff_xx = 1
	else:
		coeff_xx = 0
	Kxxy = torch.cat((Kx,Kxy),1)
	Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
	Kxyxy = torch.cat((Kxxy,Kyxy),0)
	nx = Kx.shape[0]
	ny = Ky.shape[0]

	if is_unbiased:
		xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
		yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
		# one-sample U-statistic.
		if use_1sample_U:
			xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
		else:
			xy = torch.div(torch.sum(Kxy), (nx * ny))
		mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
	else:
		xx = torch.div((torch.sum(Kx)), (nx * nx))
		yy = torch.div((torch.sum(Ky)), (ny * ny))
		# one-sample U-statistic.
		if use_1sample_U:
			xy = torch.div((torch.sum(Kxy)), (nx * ny))
		else:
			xy = torch.div(torch.sum(Kxy), (nx * ny))
		mmd2 = xx * coeff_xx - coeff_xy * xy + yy * coeff_yy
	if not is_var_computed:
		return mmd2, None, Kxyxy
	hh = Kx*coeff_xx+Ky * coeff_yy-(Kxy+Kxy.transpose(0,1))*coeff_xy/2
	V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
	V2 = (hh).sum() / (nx) / nx
	varEst = 4*(V1 - V2**2)
	if  varEst == 0.0:
		logger.warning('error_var!!'+str(V1))
	return mmd2, varEst, Kxyxy

def get_matrix_stats(matrix):
	max_val = matrix.max()       # 最大值
	min_val = matrix.min()       # 最小值
	mean_val = matrix.mean()     # 平均值
	# median_val = nmatrix) # 中位数
	info = f"Max = {max_val:.4e}, Min = {min_val:.4e}, Mean = {mean_val:.4e}"
	return info, max_val, min_val, mean_val

has_logged_xx = False

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True, is_unbiased=True, coeff_xy=2, is_yy_zero=False, is_xx_zero=False, writer=None, global_step=None):
	X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
	Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
	X_org = Fea_org[0:len_s, :] # fetch the original sample 1
	Y_org = Fea_org[len_s:, :] # fetch the original sample 2
	L = 1 # generalized Gaussian (if L>1)

	nx = X.shape[0]
	ny = Y.shape[0]
	Dxx = Pdist2(X, X)
	Dyy = Pdist2(Y, Y)
	Dxy = Pdist2(X, Y)
	Dxx_org = Pdist2(X_org, X_org)
	Dyy_org = Pdist2(Y_org, Y_org)
	Dxy_org = Pdist2(X_org, Y_org)
	Dxx_org_info, Dxx_org_max, Dxx_org_min, Dxx_org_mean = get_matrix_stats(Dxx_org)
	Dxx_info, Dxx_max, Dxx_min, Dxx_mean = get_matrix_stats(Dxx)
 
	global has_logged_xx
	if not has_logged_xx:
		logger.info("Dxx_org: " + Dxx_org_info)
		logger.info("Dxx: " + Dxx_info)
		has_logged_xx = True
  
	if writer is not None:
		writer.add_scalar(f"train/Dxx_org/max", Dxx_org_max, global_step=global_step)
		writer.add_scalar(f"train/Dxx_org/min", Dxx_org_min, global_step=global_step)
		writer.add_scalar(f"train/Dxx_org/mean", Dxx_org_mean, global_step=global_step)
		writer.add_scalar(f"train/Dxx/max", Dxx_max, global_step=global_step)
		writer.add_scalar(f"train/Dxx/min", Dxx_min, global_step=global_step)
		writer.add_scalar(f"train/Dxx/mean", Dxx_mean, global_step=global_step)
	K_Ix = torch.eye(nx).cuda()
	K_Iy = torch.eye(ny).cuda()
	if is_smooth:
		Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma)
		Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma)
		Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)
	else:
		Kx = torch.exp(-Dxx / sigma0)
		Ky = torch.exp(-Dyy / sigma0)
		Kxy = torch.exp(-Dxy / sigma0)

	return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U,is_unbiased, coeff_xy, is_yy_zero, is_xx_zero)
