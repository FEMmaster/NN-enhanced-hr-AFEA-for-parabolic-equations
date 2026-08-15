import argparse
import os
import time
import numpy as np
import scipy.io as sio

# 导入一些必要的数值计算包
from fealpy.mesh import MeshFactory as MF
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.pde.heatequation_model_2d import AllenCahn
from fealpy.functionspace import LagrangeFiniteElementSpace # 导入空间
from scipy.sparse.linalg import spsolve

import torch
from mesh_regenerator import generate_mesh_heat
from module_ML import setup_seed, NN, ELM, RBF, RBFNN

parser = argparse.ArgumentParser(description="hr自适应有限元求解AC方程")
parser.add_argument('--ns', default=10, type=int, help='空间各个方向剖分段数, 默认剖分 10 段.')
parser.add_argument('--nt', default=100, type=int, help='时间剖分段数, 默认剖分 100 段.')
parser.add_argument('--tol', default=0.5, type=float, help='自适应加密停止阈值, 默认设定为 0.05.')
parser.add_argument('--output', default='./AFEM-hr-ac', type=str, help='结果输出目录, 默认为 ./')

args = parser.parse_args()      # 解析命令行参数
ns = args.ns
nt = args.nt
tol = args.tol
output = args.output
if not os.path.exists(output):
    os.makedirs(output)

# 待求解问题设定
pde = AllenCahn()
domain = pde.domain()
epsilon = pde.diffusionCoefficient
  
# 生成网格
tmesh = UniformTimeLine(0, 0.04, nt)

# 插值模型设定
setup_seed(3407)
networks = NN((2, 32, 32, 32, 1), 'tanh', 'uniform_0.6', 'normal').to(torch.float64)
# networks = ELM([2, 1024, 2048], 'tanh', 'uniform_0.6', 'normal').to(torch.float64)
# networks = RBF([2, 3000], gamma=20.0).to(torch.float64)
# networks = RBFNN([2, 2048, 2048], gamma=60.0, act='tanh', w_init='uniform_0.6').to(torch.float64)

# 开始时间迭代
all_loss_histories = []     # 存储每个时间步的loss历史
train_iter_counts = []      # 存储每个时间步的训练迭代次数
time_step_labels = []       # 存储时间步标签
time_step_times = []        # 存储每个时间步的计算时间

total_start_time = time.time()
for j in range(0, nt+1):
    t = tmesh.current_time_level()
    dt = tmesh.current_time_step_length() #获得时间步长
    print("当前时刻是: ", t)
    
    time_step_start = time.time()
    smesh = MF.boxmesh2d([0, 1, 0, 1], nx=ns, ny=ns, meshtype='tri')
    # smesh = MF.special_boxmesh2d([0, 1, 0, 1], n=n, meshtype='rice')
    
    # 空间自适应
    lsFit = np.zeros((2, 7))
    for i in range(0, 7):
        space = LagrangeFiniteElementSpace(smesh ,p=1)
        print('    网格自由度为: ', space.number_of_global_dofs())
        
        if  j == 0:
            u_new = space.interpolation(pde.init_value)
        else:    
            M= space.mass_matrix()
            A = space.stiff_matrix()
            u1 = space.function()
            u2 = space.function()
            u_delta = space.function()
            u_old = networks.predict(smesh.node)
            u_new = space.function()
            u_new[:] = u_old
    
            #迭代过程Newton
            Error = 1        
            while Error >= 0.00000000000001:
                u1[:] = (u_new[:]**3 - u_new[:])/(epsilon**2) 
                u2[:] = (3*u_new[:]**2 - 1)/(epsilon**2)
                Mdf = space.mass_matrix(u2)
                b = space.source_vector(u1)
                # J = M+ epsilon*A*dt + Mdf *dt             # Jacobian
                # r = -((M+e*A*dt)*u_new + dt*b-M*u_old)
                
                J = M + A*dt + Mdf*dt #Jacobian
                r = -((M+A*dt)@u_new[:] + dt*b - M @ u_old[:])
                
                u_delta[:] = spsolve(J, r)
                u_new[:] = u_new[:] + u_delta[:]
                Error = space.integralalg.error(u_delta, u_delta)
                
        fname = os.path.join(output, 'test_' +str(j).zfill(5) + str(i).zfill(3) + '.vtu')
        smesh.nodedata['uh'] = u_new
        smesh.to_vtk(fname=fname)
        
        eta = space.recovery_estimate(u_new, method='area_harmonic')
        err = np.sqrt(np.sum(eta**2))
        print('    误差估计子: ', err)
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
            print("  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", flag)
        else:
            flag = 1
            
        smesh = generate_mesh_heat(smesh, eta, flag)    
        
    if j <= nt:
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
    
        loss_history = networks.fit(nodes, u_new, adam_epoch=0)
        
        all_loss_histories.extend(loss_history)          # 保存loss历史
        train_iter_counts.append(len(loss_history))      # 保存训练次数

    time_step_labels.append(t)         # 保存时间标签
    time_step_time = time.time() - time_step_start
    time_step_times.append(time_step_time) 
    print(f"  时间层 {j} 耗时: {time_step_time:.2f}秒")
    
    tmesh.advance() # 时间步进一层

total_time = time.time() - total_start_time
print(f"总耗时: {total_time:.2f}秒")