import imageio as iio_fft_res
import numpy as np
from queue import Queue
import random
from numba import jit
import warnings
from scipy import signal
from numpy.fft import fft2, ifft2, ifftshift
from PIL import Image
warnings.simplefilter('ignore')
np.set_printoptions(threshold = np.inf)

RANDOM = 0
BEST_POS = 1
BEST_PATCH = 2
w = 400
h = 400
rw = 200
rh = 200
K = 0.01

which_patch_mask = np.zeros((h, w)).astype(int)
patch_list = []

file_name = './data/green.gif'
src_img = iio_fft_res.imread(file_name).astype(np.uint8)

gen_graph = np.zeros((h, w, 4)).astype(np.uint8)
cnt = 0


class Edge:
    def __init__(self, u, v, flow):
        self.fr = u
        self.to = v
        self.flow = flow
        self.back = None

    def set_back(self, back):
        self.back = back


class Node:
    def __init__(self):
        self.edges = []
        self.cur_edge = 0
        self.depth = 0
        self.x = -1
        self.y = -1

    def add_edge(self, edge):
        self.edges += [edge]

    def set_xy(self, x, y):
        self.x = x
        self.y = y

class Graph:
    def __init__(self, node_list, pixel_cnt, S, T):
        self.nodes = node_list
        self.node_cnt = len(node_list)
        self.S = S
        self.T = T
        self.pixel_cnt = pixel_cnt
        # self.mask = np.zeros((len(self.nodes)))
        self.mask = np.zeros(self.pixel_cnt).astype(int)

    def print_node(self):
        cnt = 0
        for n in self.nodes:
            edge_cnt = len(n.edges)
            print('This is Node '+ str(cnt) + ' with ' + str(edge_cnt) + ' edges!' + '\n')
            cnt += 1
            for e in n.edges:
                print(e.fr, e.to, e.flow)
            print('....................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def dfs(self, o, remain_f):
        if o == self.T:
            return remain_f
        node = self.nodes[o]
        remain_f_ = remain_f
        i = node.cur_edge
        while i < len(node.edges):
            edge = node.edges[i]
            node.cur_edge = max(i, node.cur_edge)
            y = edge.to
            f = edge.flow
            if f > 0 and node.depth == self.nodes[y].depth + 1:
                cur_flow = self.dfs(int(y), int(min(remain_f, f)))

                edge.flow -= cur_flow
                edge.back.flow += cur_flow

                remain_f -= cur_flow
                if remain_f == 0:
                    break

            i += 1
        return remain_f_ - remain_f

    def bfs(self):
        one_q = Queue()
        one_q.put(self.T)
        for node in self.nodes:
            node.depth = 0
            node.cur_edge = 0
        while not one_q.empty():
            node = self.nodes[one_q.get()]
            for edge in node.edges:
                y = edge.to
                nt = self.nodes[y]
                if nt.depth == 0 and y != self.T and edge.back.flow > 0:
                    nt.depth = node.depth + 1
                    one_q.put(edge.to)
        return self.nodes[self.S].depth > 0

    def dinic(self):
        total_flow = 0
        dfs_times = 0
        while self.bfs():
            flow = self.dfs(self.S, 2 ** 50)
            total_flow += flow
            dfs_times += 1

    def graph_cut(self):
        self.dinic()
        self.bfs()
        for i in range(self.pixel_cnt):
            node = self.nodes[i]
            if node.depth > 0:
                self.mask[i] = 1
        return self.mask


class Patch:
    def __init__(self, patch_pos, patch_h, patch_w):
        self.in_patch_pos = (0, 0)
        self.patch_pos = patch_pos
        self.patch_h = patch_h
        self.patch_w = patch_w

    def set_patch_place(self, in_patch_pos):
        self.in_patch_pos = in_patch_pos

    def if_point_in_patch(self, pos_x, pos_y):
        return self.patch_pos[0] <= pos_x < (self.patch_pos[0] + self.patch_h) and self.patch_pos[1] <= pos_y < (
                self.patch_pos[1] + patch_w)

    # pos_x pos_y是这个gen_img中的坐标
    def gen_pic_pos_to_get_pixel(self, pos_x, pos_y):
        return src_img[self.in_patch_pos[0] + pos_x - self.patch_pos[0]][
            self.in_patch_pos[1] + pos_y - self.patch_pos[1]]


# i,j 在patch1 而 x,y 分给了patch2
#@jit
def seam_cost(patch1, patch2, i, j, x, y):
    return int(np.average(np.abs(patch1.gen_pic_pos_to_get_pixel(i, j) - patch2.gen_pic_pos_to_get_pixel(i, j))) \
           + np.average(np.abs(patch1.gen_pic_pos_to_get_pixel(x, y) - patch2.gen_pic_pos_to_get_pixel(x, y))))


# 割点时的梯度能量值
def calc_gradient(patch, x, y, direction):
    if direction == True and patch.if_point_in_patch(x - 1, y) and patch.if_point_in_patch(x + 1, y):
        t = np.average(np.abs(patch.gen_pic_pos_to_get_pixel(x - 1, y) - patch.gen_pic_pos_to_get_pixel(x + 1, y)))
        return t
    elif direction == False and patch.if_point_in_patch(x, y - 1) and patch.if_point_in_patch(x, y + 1):
        t = np.average(np.abs(patch.gen_pic_pos_to_get_pixel(x, y - 1) - patch.gen_pic_pos_to_get_pixel(x, y + 1)))
        return t
    else:
        return 1

# pos_situation: 画布上有哪些点是空的（0）和已经画上的（1） 返回前缀和
#@jit
def calc_presum(pos_situation):
    for i in range(1, pos_situation.shape[1]):
        pos_situation[0][i] = pos_situation[0][i - 1] + pos_situation[0][i]
    for i in range(1, pos_situation.shape[0]):
        pos_situation[i][0] = pos_situation[i - 1][0] + pos_situation[i][0]

    for i in range(1, pos_situation.shape[0]):
        for j in range(1, pos_situation.shape[1]):
            pos_situation[i][j] += pos_situation[i - 1][j] + pos_situation[i][j - 1] - pos_situation[i - 1][j - 1]

    return pos_situation

# pos_presum 是画布上i, j到0, 0有多少已经画上的点 现在求patch在每个i, j会覆盖多少已经画上的点
def calc_patch_situation(pos_presum):
    patch_situation = np.zeros(pos_presum.shape).astype(int)
    for i in range(0, pos_presum.shape[0]):
        for j in range(0, pos_presum.shape[1]):
            end_x = min(i + src_img.shape[0] - 1, gen_graph.shape[0] - 1)
            end_y = min(j + src_img.shape[1] - 1, gen_graph.shape[1] - 1)
            patch_situation[i][j] = pos_presum[end_x][end_y]
            if i > 0:
                patch_situation[i][j] -= pos_presum[i - 1][end_y]
            if j > 0:
                patch_situation[i][j] -= pos_presum[end_x][j - 1]
            if i > 0 and j > 0:
                patch_situation[i][j] += pos_presum[i - 1][j - 1]
    return patch_situation


# 在patch_x patch_y里放一个形状为patch_h, patch_w的patch，会不会覆盖到空点和已有点 用前缀和判断
#@jit
def pd_ptvalid(pos_presum, patch_x, patch_y, patch_h, patch_w, blank_pixel):
    new_pixel, total_pixel = get_new_pixel(pos_presum, patch_x, patch_y, patch_h, patch_w)
    s = total_pixel - new_pixel
    if blank_pixel <= 0.05 * total_pixel:
        return 0 < s < total_pixel
    if 0.05 * total_pixel < blank_pixel <= 0.4 * total_pixel:
        return 0.01 * total_pixel < new_pixel < 0.4 * total_pixel
    if blank_pixel > 0.4 * total_pixel:
        return 0.4 * total_pixel < new_pixel < 0.8 * total_pixel

def get_new_pixel(pos_presum, patch_x, patch_y, patch_h, patch_w):
    end_x = min(pos_presum.shape[0] - 1, patch_x + patch_h - 1)
    end_y = min(pos_presum.shape[1] - 1, patch_y + patch_w - 1)
    s = pos_presum[end_x][end_y]
    if patch_x > 0:
        s -= pos_presum[patch_x - 1][end_y]
    if patch_y > 0:
        s -= pos_presum[end_x][patch_y - 1]
    if patch_x > 0 and patch_y > 0:
        s += pos_presum[patch_x - 1][patch_y - 1]

    total_pixel = (end_x - patch_x + 1) * (end_y - patch_y + 1)
    new_pixel = total_pixel - s

    return new_pixel, total_pixel

# 在画布上放第一块，这一块来自于源图像的in_patch_x, in_patch_y, 形状是patch_h, patch_w，放在0 0处
def give_first_patch(patch):
    for i in range(0, patch.patch_h):
        for j in range(0, patch.patch_w):
            gen_graph[i][j] = patch.gen_pic_pos_to_get_pixel(i, j)
            which_patch_mask[i][j] = 1

# 给patch在画布里选位置，这里的patch都是整块，故in_patch_xy = 0。ret patch_x patch_y
#@jit
def choose_pos(mode):
    pos_list = []
    pos_score = []
    pos_situation = np.greater(which_patch_mask, 0).astype(int)
    pos_presum = calc_presum(pos_situation)
    patch_situation = calc_patch_situation(pos_presum)

    blank_pixel = np.sum(np.equal(which_patch_mask, 0).astype(int))

    max_pos = (-1, -1)
    max_blank = -1
    if mode == RANDOM:
        for i in range(gen_graph.shape[0]):
            for j in range(gen_graph.shape[1]):
                if pd_ptvalid(pos_presum, i, j, src_img.shape[0], src_img.shape[1], blank_pixel):
                    pos_list += [(i, j)]
                else:
                    new_pixel, total_pixel = get_new_pixel(pos_presum, i, j, src_img.shape[0], src_img.shape[1])
                    if new_pixel > max_blank:
                        max_pos = (i, j)
                        max_blank = new_pixel

        if len(pos_list) == 0:
            return max_pos
        else:
            patch_pos = pos_list[random.randint(0, len(pos_list) - 1)]
            return patch_pos

    elif mode == BEST_POS:
        
        mas = np.zeros((gen_graph.shape[0] + src_img.shape[0], gen_graph.shape[1] + src_img.shape[1], 4))
        mas[:gen_graph.shape[0], :gen_graph.shape[1]] = np.greater(which_patch_mask, 0).astype(int)[:,:, None]

        inp = np.zeros((gen_graph.shape[0] + src_img.shape[0], gen_graph.shape[1] + src_img.shape[1], 4))
        inp[:gen_graph.shape[0], :gen_graph.shape[1]] = gen_graph
        inp_2 = inp ** 2

        oup = np.zeros((gen_graph.shape[0] + src_img.shape[0], gen_graph.shape[1] + src_img.shape[1], 4))
        one = np.zeros((gen_graph.shape[0] + src_img.shape[0], gen_graph.shape[1] + src_img.shape[1], 4))
        oup[: src_img.shape[0], : src_img.shape[1]] = np.flip(src_img, axis=(0, 1))
        one[: src_img.shape[0], : src_img.shape[1]] = 1
        oup_2 = oup ** 2

        i_complex = fft2(inp, axes=(0, 1))
        i2_complex = fft2(inp_2, axes=(0, 1))
        m_complex = fft2(mas, axes=(0, 1))
        o2_complex = fft2(oup_2, axes=(0, 1))
        o_complex = fft2(oup, axes=(0, 1))
        one_complex = fft2(one, axes=(0, 1))

        i2_fft_res = ifft2(i2_complex * one_complex, axes=(0, 1)).astype(np.float64)[src_img.shape[0] - 1: src_img.shape[0] - 1 + gen_graph.shape[0], src_img.shape[1] - 1: src_img.shape[1] - 1 + gen_graph.shape[1]]
        o2_fft_res = ifft2(m_complex * o2_complex, axes=(0, 1)).astype(np.float64)[src_img.shape[0] - 1: src_img.shape[0] - 1 + gen_graph.shape[0], src_img.shape[1] - 1: src_img.shape[1] - 1 + gen_graph.shape[1]]
        io_fft_res = ifft2(i_complex * o_complex, axes=(0, 1)).astype(np.float64)[src_img.shape[0] - 1: src_img.shape[0] - 1 + gen_graph.shape[0], src_img.shape[1] - 1: src_img.shape[1] - 1 + gen_graph.shape[1]]

        cost = np.abs(np.sum(i2_fft_res[:gen_graph.shape[0], :gen_graph.shape[1]] + o2_fft_res - 2 * io_fft_res, axis=(-1)))
        cost = cost / (patch_situation[:, :] + 1)
        
        k_sigma = K * src_img.var()
        
        Cscore = np.exp(- cost / k_sigma)
        pos_sum = 0.0

        for i in range(gen_graph.shape[0]):
            for j in range(gen_graph.shape[1]):
                if pd_ptvalid(pos_presum, i, j, src_img.shape[0], src_img.shape[1], blank_pixel):
                    pos_list += [(i, j)]
                    pos_sum += Cscore[i][j]
                    pos_score += [pos_sum]

                else:
                    new_pixel, total_pixel = get_new_pixel(pos_presum, i, j, src_img.shape[0], src_img.shape[1])
                    if new_pixel > max_blank:
                        max_pos = (i, j)
                        max_blank = new_pixel

        if len(pos_list) == 0:
            return max_pos
        else:
            chosen_p = random.random() * pos_sum

            for i in range(len(pos_list)):
                if chosen_p < pos_score[i]:
                    return pos_list[i]

exist_mask = np.greater(which_patch_mask, 0).astype(int)[:,:]
# 选好了patch在画布上的位置，接下来选择原图中哪一部分当作patch
def choose_inpatch_pos(gen_graph, which_patch_mask, patch):
    pos_situation = np.greater(which_patch_mask, 0).astype(int)
    pos_presum = calc_presum(pos_situation)
    patch_situation = calc_patch_situation(pos_presum)
    exist_mask = np.greater(which_patch_mask, 0).astype(int)[:,:]
    
    mas = np.zeros((patch.patch_h + src_img.shape[0] - 1, patch.patch_w + src_img.shape[1] - 1, 4))
    inp = np.zeros((patch.patch_h + src_img.shape[0] - 1, patch.patch_w + src_img.shape[1] - 1, 4))
    mas[: patch.patch_h, : patch.patch_w] = exist_mask[patch.patch_pos[0] : patch.patch_pos[0] + patch.patch_h :, patch.patch_pos[1] : patch.patch_pos[1] + patch.patch_w :, None]
    inp[: patch.patch_h, : patch.patch_w] = gen_graph[patch.patch_pos[0] : patch.patch_pos[0] + patch.patch_h :, patch.patch_pos[1] : patch.patch_pos[1] + patch.patch_w :]
    i2 = inp ** 2

    oup = np.zeros((patch.patch_h + src_img.shape[0] - 1, patch.patch_w + src_img.shape[1] - 1, 4))
    one = np.zeros((patch.patch_h + src_img.shape[0] - 1, patch.patch_w + src_img.shape[1] - 1, 4))
    oup[: src_img.shape[0], : src_img.shape[1]] = np.flip(src_img, axis=(0,1))
    one[: src_img.shape[0], : src_img.shape[1]] = 1
    o2 = np.square(oup)

    i_complex = fft2(inp, axes=(0, 1))
    m_complex = fft2(mas, axes=(0, 1))
    o2_complex = fft2(o2, axes=(0, 1))
    o_complex = fft2(oup, axes=(0, 1))

    i2_fft_res = np.sum(i2)
    o2_fft_res = ifft2(m_complex * o2_complex, axes=(0, 1)).astype(np.float64)
    io_fft_res = ifft2(i_complex * o_complex, axes=(0, 1)).astype(np.float64)
    cost = np.sum(i2_fft_res + o2_fft_res - 2 * io_fft_res, axis=(-1))

    k_sigma = K * src_img.var()
    prob = np.exp(- np.abs(cost) / k_sigma)

    in_patch_pos_probsum = np.zeros((w - 1, h - 1))
    pos_prob_sum = 0

    for i in range(0, src_img.shape[0] - 1 - patch.patch_h):
        for j in range(0, src_img.shape[1] - 1 - patch.patch_w):
            pos_prob_sum += prob[i][j]
            in_patch_pos_probsum[i][j] = pos_prob_sum

    chosen_p = random.random() * pos_prob_sum
    for i in range(0, src_img.shape[0] - 1 - patch.patch_h):
        for j in range(0, src_img.shape[1] - 1 - patch.patch_w):
            if chosen_p < in_patch_pos_probsum[i][j]:
                return (i, j)

    return (random.randint(0, src_img.shape[0] - 2 - patch.patch_h), random.randint(0, src_img.shape[1] - 2 - patch.patch_w))


# 给图中节点连边
def make_bilink_edge(node_list, i, j, flow_front, flow_back):
    edge1_front = Edge(i, j, flow_front)
    edge1_back = Edge(j, i, flow_back)
    edge1_back.set_back(edge1_front)
    edge1_front.set_back(edge1_back)

    node_list[i].add_edge(edge1_front)
    node_list[j].add_edge(edge1_back)

# 确定割，更新gen_graph和which_patch_mask
def cut_patch(patch, patch_number):
    start_x = patch.patch_pos[0]
    start_y = patch.patch_pos[1]
    node_list = []
    node_cnt = 0
    node_idx_map = np.zeros([patch.patch_h, patch.patch_w]).astype(int)
    for i in range(start_x, min(start_x + patch.patch_h, gen_graph.shape[0])):
        for j in range(start_y, min(start_y + patch.patch_w, gen_graph.shape[1])):
            if which_patch_mask[i][j] > 0:
                node_list += [Node()]
                node_list[-1].set_xy(i, j)
                node_idx_map[i - start_x][j - start_y] = node_cnt
                node_cnt += 1

    S = node_cnt
    node_list += [Node(), Node()]
    T = len(node_list) - 1

    flag = False
    for i in range(start_x, min(start_x + patch.patch_h, gen_graph.shape[0])):
        for j in range(start_y, min(start_y + patch.patch_w, gen_graph.shape[1])):
        	if which_patch_mask[i][j] == 0:
        		flag = True

    for i in range(start_x, min(start_x + patch.patch_h, gen_graph.shape[0])):
        for j in range(start_y, min(start_y + patch.patch_w, gen_graph.shape[1])):
            if which_patch_mask[i][j] > 0:
                if j > start_y and which_patch_mask[i][j - 1] > 0:
                    if which_patch_mask[i][j - 1] != which_patch_mask[i][j]:
                        patch_idx_A1 = which_patch_mask[i][j - 1]
                        patch_idx_A2 = which_patch_mask[i][j]
                        if patch_list[patch_idx_A1 - 1].if_point_in_patch(i, j) and \
                                patch_list[patch_idx_A2 - 1].if_point_in_patch(i, j - 1):
                            # Add a new node
                            node_list += [Node()]
                            flow_A1_Node = seam_cost(patch_list[patch_idx_A1 - 1], patch, i, j - 1, i, j)
                            flow_A2_Node = seam_cost(patch, patch_list[patch_idx_A2 - 1], i, j - 1, i, j)
                            flow_B_Node = seam_cost(patch_list[patch_idx_A1 - 1], patch_list[patch_idx_A2 - 1], i, j - 1, i, j)
                            new_idx = len(node_list) - 1
                            #flow_A1_Node /= (calc_gradient(patch_list[patch_idx_A1 - 1], i, j - 1, False) + calc_gradient(patch, i, j - 1, False) + calc_gradient(patch_list[patch_idx_A1 - 1], i, j, False) + calc_gradient(patch, i, j, False))
                            #flow_A2_Node /= (calc_gradient(patch_list[patch_idx_A2 - 1], i, j - 1, False) + calc_gradient(patch, i, j - 1, False) + calc_gradient(patch_list[patch_idx_A2 - 1], i, j, False) + calc_gradient(patch, i, j, False))
                            #flow_B_Node /= (calc_gradient(patch_list[patch_idx_A1 - 1], i, j - 1, False) + calc_gradient(patch_list[patch_idx_A2 - 1], i, j - 1, False) + calc_gradient(patch_list[patch_idx_A1 - 1], i, j, False) + calc_gradient(patch_list[patch_idx_A2 - 1], i, j, False))
                            
                            flow_A1_Node = round(flow_A1_Node)
                            flow_A2_Node = round(flow_A2_Node)
                            flow_B_Node = round(flow_B_Node)

                            make_bilink_edge(node_list, node_idx_map[i - start_x][j - 1 - start_y], new_idx, flow_A1_Node, flow_A1_Node)
                            make_bilink_edge(node_list, node_idx_map[i - start_x][j - start_y], new_idx, flow_A2_Node, flow_A2_Node)
                            make_bilink_edge(node_list, new_idx, T, flow_B_Node, 0)

                    else:
                        patch_idx = which_patch_mask[i][j]
                        flow = seam_cost(patch_list[patch_idx - 1], patch, i, j - 1, i, j)
                        #flow /= (calc_gradient(patch_list[patch_idx - 1], i, j - 1, False) + calc_gradient(patch_list[patch_idx - 1], i, j, False) + calc_gradient(patch, i, j - 1, False) + calc_gradient(patch, i, j, False))
                        flow = round(flow)
                        make_bilink_edge(node_list, node_idx_map[i - start_x][j - 1 -start_y], node_idx_map[i - start_x][j - start_y], flow, flow)

                if i > start_x and which_patch_mask[i - 1][j] > 0:
                    if which_patch_mask[i - 1][j] != which_patch_mask[i][j]:
                        patch_idx_A1 = which_patch_mask[i - 1][j]
                        patch_idx_A2 = which_patch_mask[i][j]
                        if patch_list[patch_idx_A1 - 1].if_point_in_patch(i, j) and \
                                patch_list[patch_idx_A2 - 1].if_point_in_patch(i - 1, j):
                            # Add a new node
                            node_list += [Node()]
                            flow_A1_Node = seam_cost(patch_list[patch_idx_A1 - 1], patch, i - 1, j, i, j)
                            flow_A2_Node = seam_cost(patch, patch_list[patch_idx_A2 - 1], i - 1, j, i, j)
                            flow_B_Node = seam_cost(patch_list[patch_idx_A1 - 1], patch_list[patch_idx_A2 - 1], i - 1, j, i, j)

                            #flow_A1_Node /= (calc_gradient(patch_list[patch_idx_A1 - 1], i - 1, j, True) + calc_gradient(patch, i - 1, j, True) + calc_gradient(patch_list[patch_idx_A1 - 1], i, j, True) + calc_gradient(patch, i, j, True))
                            #flow_A2_Node /= (calc_gradient(patch_list[patch_idx_A2 - 1], i - 1, j, True) + calc_gradient(patch, i - 1, j, True) + calc_gradient(patch_list[patch_idx_A2 - 1], i, j, True) + calc_gradient(patch, i, j, True))
                            #flow_B_Node /= (calc_gradient(patch_list[patch_idx_A1 - 1], i - 1, j, True) + calc_gradient(patch_list[patch_idx_A2 - 1], i - 1, j, True) + calc_gradient(patch_list[patch_idx_A1 - 1], i, j, True) + calc_gradient(patch_list[patch_idx_A2 - 1], i, j, True))
                            
                            flow_A1_Node = round(flow_A1_Node)
                            flow_A2_Node = round(flow_A2_Node)
                            flow_B_Node = round(flow_B_Node)
                            new_idx = len(node_list) - 1

                            make_bilink_edge(node_list, node_idx_map[i - 1 - start_x][j - start_y], new_idx, flow_A1_Node, flow_A1_Node)
                            make_bilink_edge(node_list, node_idx_map[i - start_x][j - start_y], new_idx, flow_A2_Node, flow_A2_Node)
                            make_bilink_edge(node_list, new_idx, T, flow_B_Node, 0)

                    else:
                        patch_idx = which_patch_mask[i][j]
                        flow = seam_cost(patch_list[patch_idx - 1], patch, i - 1, j, i, j)
                        #flow /= (calc_gradient(patch_list[patch_idx - 1], i - 1, j, True) + calc_gradient(patch_list[patch_idx - 1], i, j, True) + calc_gradient(patch, i - 1, j, True) + calc_gradient(patch, i, j, True))
                        flow = round(flow)
                        
                        make_bilink_edge(node_list, node_idx_map[i - 1 - start_x][j - start_y], node_idx_map[i - start_x][j - start_y], flow, flow)
                    

                for pos in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    # 是原有的覆盖点，不是新patch的点
                    # 是新patch的点，不是原有的覆盖点
                    if 0 <= pos[0] < gen_graph.shape[0] and 0 <= pos[1] < gen_graph.shape[1]:
                        if which_patch_mask[pos[0]][pos[1]] == 0 and patch.if_point_in_patch(pos[0], pos[1]):
                            make_bilink_edge(node_list, node_idx_map[i - start_x][j - start_y], T, 2 ** 50, 0)
                            break
                        if which_patch_mask[pos[0]][pos[1]] > 0 and not patch.if_point_in_patch(pos[0], pos[1]):
                            make_bilink_edge(node_list, S, node_idx_map[i - start_x][j - start_y], 2 ** 50, 0)
                            break
    
    # surrounded region
    if flag == False and patch.patch_h > 5 and patch.patch_w > 5:
        pick_rx = random.randint(start_x + 2, min(start_x + patch.patch_h, gen_graph.shape[0]) - 2)
        pick_ry = random.randint(start_y + 2, min(start_y + patch.patch_w, gen_graph.shape[1]) - 2)
        make_bilink_edge(node_list, node_idx_map[pick_rx - start_x][pick_ry - start_y], T, 2 ** 50, 0)

    G = Graph(node_list, node_cnt, S, T)
    #G.print_node()
    patch_mask = G.graph_cut()
    for i in range(node_cnt):
        if patch_mask[i] == 1:
            pixel_x = node_list[i].x
            pixel_y = node_list[i].y
            assert(which_patch_mask[pixel_x][pixel_y] > 0)
            gen_graph[pixel_x][pixel_y] = patch.gen_pic_pos_to_get_pixel(pixel_x, pixel_y)
            which_patch_mask[pixel_x][pixel_y] = patch_number


    for i in range(start_x, min(start_x + patch.patch_h, gen_graph.shape[0])):
        for j in range(start_y, min(start_y + patch.patch_w, gen_graph.shape[1])):
            if which_patch_mask[i][j] == 0:
                gen_graph[i][j] = patch.gen_pic_pos_to_get_pixel(i, j)
                which_patch_mask[i][j] = patch_number


# =========================================================================================

mode = BEST_POS
right_below_x = -1
right_below_y = -1
while True:
    cnt += 1
    print(cnt)

    if cnt == 1:
        if mode == RANDOM or mode == BEST_POS:
            first_patch = Patch((0, 0), src_img.shape[0], src_img.shape[1])
            give_first_patch(first_patch)
            patch_list += [first_patch]
        else:
            first_patch = Patch((0, 0), int(src_img.shape[0] * 0.6), int(src_img.shape[1] * 0.6))
            give_first_patch(first_patch)
            patch_list += [first_patch]
            right_below_x = int(src_img.shape[0] * 0.6)
            right_below_y = int(src_img.shape[1] * 0.6)
    else:
        if mode == RANDOM or mode == BEST_POS:
            patch_h, patch_w, _ = src_img.shape
            patch_pos = choose_pos(mode)
            patch_h = min(patch_h, gen_graph.shape[0] - patch_pos[0])
            patch_w = min(patch_w, gen_graph.shape[1] - patch_pos[1])
            new_patch = Patch(patch_pos, patch_h, patch_w)
            patch_number = len(patch_list) + 1
            cut_patch(new_patch, patch_number)
            patch_list += [new_patch]

            if cnt % 5 == 0:
                K *= 1.2
            #img = Image.fromarray(gen_graph.astype(np.uint8))
            
            #img.show('output_tmp.gif')
        else:
            patch_h = int(src_img.shape[0] * 0.6)
            patch_w = int(src_img.shape[1] * 0.6)
            if right_below_y < gen_graph.shape[1] - 1:
                patch_x = right_below_x - patch_h
                patch_y = min(gen_graph.shape[1] - 1, right_below_y - patch_w + int(random.randint(4, 8) / 10 * patch_w))
            else:
                patch_x = min(gen_graph.shape[0] - 1, right_below_x - patch_h + int(random.randint(4, 8) / 10 * patch_h))
                patch_y = 0
            patch_h = int(min(patch_h, gen_graph.shape[0] - patch_x))
            patch_w = int(min(patch_w, gen_graph.shape[1] - patch_y))
            right_below_x = patch_x + patch_h
            right_below_y = patch_y + patch_w
            new_patch = Patch((patch_x, patch_y), patch_h, patch_w)
            patch_number = len(patch_list) + 1
            in_patch_pos = choose_inpatch_pos(gen_graph, which_patch_mask, new_patch)
            new_patch.set_patch_place(in_patch_pos)
            cut_patch(new_patch, patch_number)

            patch_list += [new_patch]

            if cnt % 5 == 0:
                K *= 1.2

    if np.sum(np.equal(which_patch_mask[:rh, :rw], 0)) == 0:
        break

print('================================= Filled Complete ================================================')

'''
cnt = (gen_graph.shape[0] * gen_graph.shape[1]) / (src_img.shape[0] * src_img.shape[1])
while cnt > 0:
    cnt -= 1
    patch_x = random.randint(0, gen_graph.shape[0] - 1)
    patch_y = random.randint(0, gen_graph.shape[1] - 1)
    patch_pos = (patch_x, patch_y)
    patch_h = min(src_img.shape[0], gen_graph.shape[0] - patch_pos[0])
    patch_w = min(src_img.shape[1], gen_graph.shape[1] - patch_pos[1])
    new_patch = Patch(patch_pos, patch_h, patch_w)
    patch_number = len(patch_list) + 1
    cut_patch(new_patch, patch_number)
    patch_list += [new_patch]
'''
iio_fft_res.imwrite("output_green_bestpos.gif", gen_graph[:rh, :rw])
