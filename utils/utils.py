"""
Code borrowed from Xinshuo_PyToolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox
"""

import os
import shutil
import torch
import numpy as np
import random
import time
import copy
import glob, glob2
from torch import nn

"""
    AverageMeter 用于计算和存储数值的平均值及当前值，常用于模型训练过程中监控性能指标。
    reset 方法初始化所有属性。
    update 方法用于更新当前值、总和和计数，从而动态计算平均值。
"""
class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0    
    self.list = list()
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count  
    self.list.append(val)


"""
    isnparray, isinteger, isfloat, isscalar, islogical, isstring, islist：这些函数用于检查变量的数据类型，
    分别判断是否为 NumPy 数组、整数、浮点数、标量、布尔值、字符串和列表。
    它们通过使用 isinstance 函数实现基本的类型检查，确保传入数据类型符合预期。
"""
def isnparray(nparray_test):
	return isinstance(nparray_test, np.ndarray)


def isinteger(integer_test):
	if isnparray(integer_test): return False
	try: return isinstance(integer_test, int) or int(integer_test) == integer_test
	except ValueError: return False
	except TypeError: return False


def isfloat(float_test):
	return isinstance(float_test, float)


def isscalar(scalar_test):
	try: return isinteger(scalar_test) or isfloat(scalar_test)
	except TypeError: return False


def islogical(logical_test):
	return isinstance(logical_test, bool)

    
def isstring(string_test):
    return isinstance(string_test, str)


def islist(list_test):
	return isinstance(list_test, list)

"""
    convert_secs2time 将秒数转换为小时、分钟、秒的格式，返回一个字符串，如 [1:23:45]。
"""
def convert_secs2time(seconds):
    '''
    format second to human readable way
    '''
    assert isscalar(seconds), 'input should be a scalar to represent number of seconds'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '[%d:%02d:%02d]' % (h, m, s)


"""
    get_timestring 返回当前时间的字符串格式，如 20231025_14h30m15s，便于命名文件或日志。
"""
def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')


"""
    recreate_dirs 函数重建目录，即删除现有的同名目录并创建一个新的空目录。它接受多个目录路径作为参数。
"""
def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


"""
    is_path_valid 检查路径是否为有效字符串，若是空字符串或非字符串，则返回 False
"""
def is_path_valid(pathname):
	try:  
		if not isstring(pathname) or not pathname: return False
	except TypeError: return False
	else: return True


"""
    is_path_creatable 检查指定路径的父级目录是否存在且可写入，以便在路径上创建新文件夹或文件。
"""
def is_path_creatable(pathname):
	'''
	if any previous level of parent folder exists, returns true
	'''
	if not is_path_valid(pathname): return False
	pathname = os.path.normpath(pathname)
	pathname = os.path.dirname(os.path.abspath(pathname))

	# recursively to find the previous level of parent folder existing
	while not is_path_exists(pathname):     
		pathname_new = os.path.dirname(os.path.abspath(pathname))
		if pathname_new == pathname: return False
		pathname = pathname_new
	return os.access(pathname, os.W_OK)


"""
    判断路径是否存在
"""
def is_path_exists(pathname):
	try: return is_path_valid(pathname) and os.path.exists(pathname)
	except OSError: return False


def is_path_exists_or_creatable(pathname):
	try: return is_path_exists(pathname) or is_path_creatable(pathname)
	except OSError: return False


"""
    isfile 检查路径是否为有效文件路径，通过检查文件名和扩展名是否存在来实现。
"""
def isfile(pathname):
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) > 0
	else: return False


"""
    isfolder 检查路径是否为有效文件夹路径，不允许路径名中包含扩展名。
"""
def isfolder(pathname):
	'''
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	'''
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		if pathname == './': return True
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) == 0
	else: return False


"""
    mkdir_if_missing 检查路径是否存在，不存在时创建该路径，确保文件夹准备好用于写入文件。
"""
def mkdir_if_missing(input_path):
    folder = input_path if isfolder(input_path) else os.path.dirname(input_path)
    os.makedirs(folder, exist_ok=True)


"""
    safe_list 和 safe_path 确保输入的列表或路径有效，并通过 copy 模块创建一个安全副本，以防止修改原数据。
"""
def safe_list(input_data, warning=True, debug=True):
	'''
	copy a list to the buffer for use
	parameters:
		input_data:		a list
	outputs:
		safe_data:		a copy of input data
	'''
	if debug: assert islist(input_data), 'the input data is not a list'
	safe_data = copy.copy(input_data)
	return safe_data

def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'
    parameters:
    	input_path:		a string
    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


"""
    prepare_seed 设置 NumPy、Python 和 PyTorch 的随机种子，用于实验可重复性。
    ******************************************************************
"""
def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)

"""
    initialize_weights 用于初始化模型的权重，针对不同的层使用适当的初始化方法，例如 Conv2d 层的 kaiming_normal_ 方法。
"""
def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)


"""
    print_log 用于打印并记录日志信息，支持同一行输出和写入文件。
"""
def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file

	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	if same_line: log.write('{}'.format(print_str))
	else: log.write('{}\n'.format(print_str))
	log.flush()


"""
    find_unique_common_from_lists 查找两个列表的公共元素，返回这些公共元素及其在两个列表中的索引。
"""
def find_unique_common_from_lists(input_list1, input_list2, warning=True, debug=True):
	'''
	find common items from 2 lists, the returned elements are unique. repetitive items will be ignored
	if the common items in two elements are not in the same order, the outputs follows the order in the first list

	parameters:
		input_list1, input_list2:		two input lists

	outputs:
		list_common:	a list of elements existing both in list_src1 and list_src2	
		index_list1:	a list of index that list 1 has common items
		index_list2:	a list of index that list 2 has common items
	'''
	input_list1 = safe_list(input_list1, warning=warning, debug=debug)
	input_list2 = safe_list(input_list2, warning=warning, debug=debug)

	common_list = list(set(input_list1).intersection(input_list2))
	
	# find index
	index_list1 = []
	for index in range(len(input_list1)):
		item = input_list1[index]
		if item in common_list:
			index_list1.append(index)

	index_list2 = []
	for index in range(len(input_list2)):
		item = input_list2[index]
		if item in common_list:
			index_list2.append(index)

	return common_list, index_list1, index_list2


"""
    load_txt_file 加载文本文件并返回内容和行数.
"""
def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safe_path(file_path)
    if debug: assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file: data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines


"""
    load_list_from_folder 从指定文件夹加载文件或子文件夹列表，可选择递归加载到特定深度。
"""

def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    '''
    load a list of files or folders from a system path

    parameters:
        folder_path:    root to search 
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels till to the input depth

    outputs:
        fulllist:       a list of elements
        num_elem:       number of the elements
    '''
    folder_path = safe_path(folder_path)
    if debug: assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path): 
        print('the input folder does not exist\n')
        return [], 0
    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter): ext_filter = [ext_filter]                               # convert to a list
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist: file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem


# def abs_to_relative(abs_traj):
#     """
#     将绝对坐标轨迹转换为相对位移轨迹
#     输入:
#         abs_traj: 绝对轨迹，形状为 (seq_len, batch, num_agents, 2)
#     输出:
#         rel_traj: 相对轨迹，形状为 (seq_len, batch, num_agents, 2)
#     """
#     # 计算相邻时间步之间的差分
#     # abs_traj[1:] 的形状为 (seq_len-1, batch, num_agents, 2)
#     # abs_traj[:-1] 的形状也为 (seq_len-1, batch, num_agents, 2)
#     deltas = abs_traj[1:] - abs_traj[:-1]
#
#     # 第一帧补零，形状为 (1, batch, num_agents, 2)
#     zeros = torch.zeros(1, *abs_traj.shape[1:], device=abs_traj.device, dtype=abs_traj.dtype)
#
#     # 拼接后得到相对轨迹，形状为 (seq_len, batch, num_agents, 2)
#     rel_traj = torch.cat([zeros, deltas], dim=0)
#
#     return rel_traj


# def abs_to_relative(abs_traj):
#     """
#     将绝对坐标轨迹转换为相对位移轨迹
#     输入:
#         abs_traj: 绝对轨迹，形状为 (batch, seq_len, num_agents, 2)
#     输出:
#         rel_traj: 相对轨迹，形状为 (seq_len, batch*num_agents, 2)
#     """
#     # 将时间维度移到第一位
#     # (batch, seq_len, num_agents, 2) -> (seq_len, batch, num_agents, 2)
#     abs_traj = abs_traj.permute(1, 0, 2, 3)
#
#     # 将 batch 与 num_agents 合并为新的 batch 维度
#     seq_len, batch, num_agents, _ = abs_traj.shape
#     abs_traj = abs_traj.reshape(seq_len, batch * num_agents, 2)
#
#     # 计算相邻时间步之间的差分，即 x_t - x_{t-1}
#     # deltas 的形状为 (seq_len-1, batch*num_agents, 2)
#     deltas = abs_traj[1:] - abs_traj[:-1]
#
#     # 第一帧补零，使得序列长度保持不变
#     zeros = torch.zeros(1, batch * num_agents, 2, device=abs_traj.device, dtype=abs_traj.dtype)
#
#     # 拼接得到相对轨迹，形状为 (seq_len, batch*num_agents, 2)
#     rel_traj = torch.cat([zeros, deltas], dim=0)
#
#     return rel_traj
def abs_to_relative(abs_traj):
    """
    将绝对坐标轨迹转换为相对位移轨迹
    Input:
        abs_traj: 绝对轨迹，形状为 (seq_len, batch, 2)
    Output:
        rel_traj: 相对轨迹，形状为 (seq_len, batch, 2)
    """
    # 调整维度为 (batch, seq_len, 2)
    abs_traj = abs_traj.permute(1, 0, 2)

    # 计算位移差：x_t - x_{t-1}
    # 结果形状 (batch, seq_len-1, 2)
    deltas = abs_traj[:, 1:] - abs_traj[:, :-1]

    # 首帧补零，保持序列长度一致
    zeros = torch.zeros_like(abs_traj[:, :1])  # 形状 (batch, 1, 2)
    rel_traj = torch.cat([zeros, deltas], dim=1)  # 形状 (batch, seq_len, 2)

    # 恢复原始维度 (seq_len, batch, 2)
    rel_traj = rel_traj.permute(1, 0, 2)

    return rel_traj


def convert_to_4d(traj, seq_start_end):
    # traj: (T, batch, feature_dim)
    if seq_start_end is None:
        # 单一轨迹场景：输出 (batch, T, 1, feature_dim)
        return traj.permute(1, 0, 2).unsqueeze(2)
    else:
        scene_inputs = []
        for (start, end) in seq_start_end:
            # 对应一个场景内的轨迹，形状 (T, num_agents, feature_dim)
            traj_scene = traj[:, start:end, :]
            # 调整为 (num_agents, T, feature_dim)，再增加一个场景维度 → (1, T, num_agents, feature_dim)
            scene_inputs.append(traj_scene.permute(1, 0, 2).unsqueeze(0))
        # 将所有场景在 batch 维度上拼接，得到 (B, T, N, feature_dim)
        return torch.cat(scene_inputs, dim=0)

