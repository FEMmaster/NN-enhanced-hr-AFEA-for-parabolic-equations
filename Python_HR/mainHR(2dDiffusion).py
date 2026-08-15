#!/usr/bin/env python3
# 
import argparse
import os
import time
import numpy as np
import scipy.io as sio
from matplotlib import rc
rc('text', usetex=True)

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.heatequation_model_2d import Rotation, Diffusion, Splitting, LShapeHeatSource, LShapeHeatSource1, LShapePeriodicHeatSource
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from scipy.sparse.linalg import spsolve
from fealpy.mesh.quality import TriRadiusRatio

import torch
from mesh_regenerator import generate_mesh_heat, generate_mesh_heat_lshape
from module_ML import setup_seed, NN, ELM, RBF, RBFNN

## 参数解析
parser = argparse.ArgumentParser(description = "hr自适应有限元求解HEAT方程")
parser.add_argument('--ns', default=22, type=int, help='空间各个方向剖分段数, 默认剖分 2 次.')
parser.add_argument('--nt', default=100, type=int, help='时间剖分段数, 默认剖分 100 段.')
parser.add_argument('--tol', default=0.5, type=float, help='自适应加密停止阈值, 默认设定为 0.05.')
parser.add_argument('--output', default='./AFEM-hr-heat', type=str, help='结果输出目录, 默认为 ./')

args = parser.parse_args()      # 解析命令行参数
ns = args.ns
nt = args.nt
tol = args.tol
output = args.output
if not os.path.exists(output):
    os.makedirs(output)

# 待求解问题设定    
pde = Diffusion()
domain = pde.domain()
c = pde.diffusionCoefficient

# 网格设定 (hr自适应空间网格剖分在循环内部)
tmesh = UniformTimeLine(0, 1, nt) # 均匀时间剖分

# 插值模型设定
# setup_seed(3407)
networks = NN((2, 40, 40, 40, 1), 'tanh', 'uniform_0.6', 'normal').to(torch.float64)
# networks = ELM([2, 1024, 2048], 'tanh', 'uniform_0.6', 'normal').to(torch.float64)
# networks = RBF([2, 3000], gamma=20.0).to(torch.float64)
# networks = RBFNN([2, 2048, 2048], gamma=60.0, act='tanh', w_init='uniform_0.6').to(torch.float64)

# 开始时间迭代
all_loss_histories = []     # 存储每个时间步的loss历史
train_iter_counts = []      # 存储每个时间步的训练迭代次数
time_step_labels = []       # 存储时间步标签
time_step_times = []        # 存储每个时间步的计算时间
degree_of_freedom = []      # 存储每个时间步最终自由度

total_start_time = time.time()
for j in range(0, nt+1):
    t = tmesh.current_time_level()
    dt = tmesh.current_time_step_length() # 时间步长
    print("当前时刻是: ", t)
    
    time_step_start = time.time()
    smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    # smesh = MF.lshape_mesh(n=3)
    # meshQuality = TriRadiusRatio()

    lsFit = np.zeros((2, 7))
    for i in range(0, 7):
        # meshQuality.show(meshQuality.quality(mesh=smesh))
        space = LagrangeFiniteElementSpace(smesh, p=1)
        print('    网格自由度为: ', space.number_of_global_dofs())
        
        if j == 0:
            uh1 = space.interpolation(pde.init_value)
        else:                           
            # 下一层时间步的有限元解
            uh1 = space.function()
            A = c*space.stiff_matrix() # 刚度矩阵
            M = space.mass_matrix() # 质量矩阵
            G = M + dt*A # 隐式迭代矩阵

            # 当前时间层的右端项
            @cartesian
            def source(p):
                return pde.source(p, t)
            F = space.source_vector(source)
            F *= dt
            F += M@networks.predict(smesh.node)

            # 当前时间层的 Dirichlet 边界条件处理
            @cartesian
            def dirichlet(p):
                return pde.dirichlet(p, t)
            bc = DirichletBC(space, dirichlet)
            GD, F = bc.apply(G, F, uh1)

            # 代数系统求解
            uh1[:] = spsolve(GD, F)
        
        # 保存网格和数值解信息
        u_exact = pde.solution(smesh.node, t)
        
        fname = os.path.join(output, 'test_' +str(j).zfill(5) + str(i).zfill(3) + '.vtu')

        smesh.nodedata['u_exact'] = u_exact
        smesh.nodedata['uh'] = uh1
        smesh.nodedata['abs_error'] = np.abs(uh1 - u_exact)
        smesh.to_vtk(fname=fname)
        
        eta = space.recovery_estimate(uh1, method='area_harmonic')
        err = np.sqrt(np.sum(eta**2))
        # smesh.nodedata['err_ZZ'] = eta
        print('      误差估计子: ', err)
        lsFit[0, i] = smesh.number_of_nodes()
        lsFit[1, i] = err
        if err < tol or i > 5:
            break
        if i == 4:
            A = np.ones((3, 2))
            b = np.log(lsFit[1, 2:5])
            A[:, 1] = np.log(lsFit[0, 2:5])
            p = np.linalg.lstsq(A, b, rcond=None)[0]
            ideaN = int(np.floor(np.exp((np.log(tol) - p[0]) / p[1])))
            flag = max(int(np.floor(np.log(ideaN / lsFit[0, i]) / np.log(2))), 1)
            print("    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", flag)
        else:
            flag = 1.1

        smesh = generate_mesh_heat(smesh, eta, flag)
        # smesh = generate_mesh_heat_lshape(smesh, eta, flag)
        
    mode = "w" if t == 0 else "a"
    with open(os.path.join(output, "result.csv"), mode) as nodeFile:
        for j in range(i + 1):
            nodeFile.write(f"{t},{j},{lsFit[0, j]},{lsFit[1, j]}\n")
                
    # # t 时间层的误差
    # @cartesian
    # def solution(p):
    #     return pde.solution(p, t)
    # @cartesian
    # def grad_solution(p):
    #     return np.array([pde.u_x(p, t), pde.u_y(p, t)])
    # error_L2 = space.integralalg.error(solution, uh1)
    # error_H1_semi = space.integralalg.error(grad_solution, uh1.grad_value)
    # error_H1 = np.sqrt(error_L2**2 + error_H1_semi**2)
    
    if j < nt:
        nodes = smesh.entity('node')
        # cells = smesh.entity('cell')
        # Lambda, Weight = get_quadrature_rule(2, 6)
        # t_gaussPoint, t_area = trans_cartesian(nodes, cells, Lambda)
        # x_gauss = t_gaussPoint[:, 0]
        # y_gauss = t_gaussPoint[:, 1]
        # aa = uh1[cells]
        # gauss_values = np.dot(aa, Lambda.T).T.reshape(-1)
        # combined_p = np.vstack((nodes, t_gaussPoint))
        # combined_v = np.hstack((uh1, gauss_values))[:, None]
    
        loss_history = networks.fit(nodes, uh1, adam_epoch=0)
        
        all_loss_histories.extend(loss_history)          # 保存loss历史
        train_iter_counts.append(len(loss_history))      # 保存训练次数
    time_step_time = time.time() - time_step_start
    print(f"  时间层 {j} 耗时: {time_step_time:.2f}秒")
    
    tmesh.advance() # 时间步进一层
    
    time_step_labels.append(t)         # 保存时间标签
    time_step_times.append(time_step_time) 
    degree_of_freedom.append(smesh.number_of_nodes()) 

total_time =  time.time() - total_start_time
print(f"总耗时: {total_time:.2f}秒")

# ========== 统计训练次数与时间步耗时（排除初始时刻） ==========

train_iter_array = np.asarray(train_iter_counts, dtype=float).flatten()
time_step_time_array = np.asarray(time_step_times, dtype=float).flatten()

# 排除第一个时刻（索引0）
train_after_first = train_iter_array[1:]
time_after_first = time_step_time_array[1:]

# 数据点个数
num_points = len(train_after_first)

# 总训练次数
total_train_iters = np.sum(train_after_first)

# 平均训练次数
mean_train_iters = np.mean(train_after_first)

# 平均耗时
avg_time_per_step = np.mean(time_after_first)

# 排除初始时刻后的总时间步耗时
total_time_after_first = np.sum(time_after_first)

print("\n" + "=" * 60)
print("统计结果（已排除初始时刻）")
print("=" * 60)
print(f"数据点个数: {num_points}")
print(f"总训练次数: {total_train_iters:.0f}")
print(f"训练次数平均值: {mean_train_iters:.1f}")
print(f"时间步平均耗时: {avg_time_per_step:.4f} 秒")
print(f"时间步总耗时: {total_time_after_first:.4f} 秒")
print("=" * 60)

sio.savemat(os.path.join(output, 'all_loss_histories.mat'), {'all_loss_histories': np.array(all_loss_histories, dtype=np.float64)})
sio.savemat(os.path.join(output, 'train_iter_counts.mat'), {'train_iter_counts': np.array(train_iter_counts, dtype=np.int32)})
sio.savemat(os.path.join(output, 'time_step_labels.mat'), {'time_step_labels': np.array(time_step_labels, dtype=np.float64)})
sio.savemat(os.path.join(output, 'time_step_times.mat'), {'time_step_times': np.array(time_step_times, dtype=np.float64)})
sio.savemat(os.path.join(output, 'degree_of_freedom.mat'), {'degree_of_freedom': np.array(degree_of_freedom, dtype=np.float64)})