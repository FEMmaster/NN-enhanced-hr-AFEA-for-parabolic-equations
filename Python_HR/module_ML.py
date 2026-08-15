import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SinActivation(nn.Module):
    def forward(self, x): return torch.sin(torch.pi * x)

class CosActivation(nn.Module):
    def forward(self, x): return torch.cos(torch.pi * x)

def get_activation_fn(act_name):
    registry = {
        'tanh':    lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
        'relu':    lambda: nn.ReLU(),
        'sin':     lambda: SinActivation(),
        'cos':     lambda: CosActivation(),
        'identity': lambda: nn.Identity()
    }
    if act_name not in registry:
        raise ValueError(f"Unsupported activation function: {act_name}")
    return registry[act_name]()  # 返回一个新的激活函数模块

def apply_weight_init(layer, method_str):
    # 拆分方法名和参数值
    parts = method_str.rsplit('_', 1)
    if len(parts) == 2:
        param = float(parts[1])           # 尝试将最后一段转为参数
        method = parts[0]                 # 只有成功转换时才拆分方法名
    else:
        param = None
        method = method_str               # 转换失败说明是完整方法名，无参数
        
    methods = {
        'normal': lambda x: nn.init.normal_(x.weight, mean=0.0, std=(param or 1.0)),
        'uniform': lambda x: nn.init.uniform_(x.weight, a=-(param or 1.0), b=(param or 1.0)),
        'xavier_uniform': lambda x: nn.init.xavier_uniform_(x.weight, gain=(param or 1.0)),
        'xavier_normal': lambda x: nn.init.xavier_normal_(x.weight, gain=(param or 1.0)),
        'kaiming_uniform': lambda x: nn.init.kaiming_uniform_(x.weight),
        'kaiming_normal': lambda x: nn.init.kaiming_normal_(x.weight),
        'orthogonal': lambda x: nn.init.orthogonal_(x.weight, gain=(param or 1.0))
    }
    if method not in methods:
        raise ValueError(f"Unsupported weight init method: {method}")
    methods[method](layer)

def apply_bias_init(layer, method):
    methods = {
        'normal': lambda x: nn.init.normal_(x.bias, mean=0.0, std=0.5),
        'uniform': lambda x: nn.init.uniform_(x.bias, a=-0.5, b=0.5),
    } 
    if method not in methods:
        raise ValueError(f"Unsupported bias init method: {method}")
    methods[method](layer)


class NN(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal', b_init='normal'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        # 创建主干网络（MLP）
        self.model = nn.Sequential()
        for i in range(len(mlp_layers) - 2):
            linear = nn.Linear(mlp_layers[i], mlp_layers[i + 1], bias=True)
            apply_weight_init(linear, w_init)
            apply_bias_init(linear, b_init)
            self.model.add_module(f'fc{i + 1}', linear)         
            self.model.add_module(f'act{i + 1}', get_activation_fn(act))
        linear = nn.Linear(mlp_layers[-2], mlp_layers[-1], bias=False)
        apply_weight_init(linear, w_init)
        self.model.add_module(f'fc{len(mlp_layers) - 1}', linear)
        self.to(self.device).double()
        
        # 初始化归一化范围
        self.X_min = None
        self.X_max = None
        self.U_min = None
        self.U_max = None

    def distance_function(self, xy):
        x, y = xy[:, 0], xy[:, 1]
        
        # 计算到正方形 [-1,1]×[-1,1] 各边界的距离
        d_left = x - (-1.0)    # 到左边界 (x=-1) 的距离
        d_right = 1.0 - x      # 到右边界 (x=1) 的距离
        d_bottom = y - (-1.0)  # 到下边界 (y=-1) 的距离
        d_top = 1.0 - y        # 到上边界 (y=1) 的距离
        
        # 取到最近边界的距离
        d_square = torch.minimum(
            torch.minimum(d_left, d_right),
            torch.minimum(d_bottom, d_top)
        )
        
        return d_square

    def forward(self, X):
        # D = self.distance_function(X)
        # D_clamped = D.clamp(min=0).unsqueeze(1)
        # return D_clamped * self.model(X)
        return self.model(X)

    def normalize(self, X, X_min, X_max):
        scale = X_max - X_min
        scale[scale == 0] = 1.0 
        return 2 * (X - X_min) / scale - 1      # 归一化到 [-1, 1]
        # return (X - X_min) / scale              # 归一化到 [0, 1]

    def denormalize(self, X, X_min, X_max):
        scale = X_max - X_min
        scale[scale == 0] = 1.0 
        return (X + 1) / 2 * scale + X_min      # 归一化到 [-1, 1]
        # return X * scale + X_min                # 归一化到 [0, 1]
    
    def fit(self, X, U, adam_epoch=2000, lbfgs_max_iter=10000):
        """训练网络"""
        self.train()
         
        # 计算归一化范围并归一化输入和输出
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.U_min = U.min(axis=0)
        self.U_max = U.max(axis=0)

        X = self.normalize(X, self.X_min, self.X_max)
        U = self.normalize(U, self.U_min, self.U_max)

        X = torch.from_numpy(X).to(self.device).double()
        U = torch.from_numpy(U).to(self.device).double()

        # 损失函数
        criterion = nn.MSELoss()
        loss_history = []
        
        # ADAM优化器
        optimizer_adam = optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(adam_epoch):
            self.zero_grad()
            U_nn = self(X)
            
            # 计算损失
            if U.shape[-1] == U_nn.shape[-1] == 2:  # 向量情况
                loss = criterion(U_nn[:, 0], U[:, 0]) + criterion(U_nn[:, 1], U[:, 1])
            else:  # 标量情况
                loss = criterion(U_nn, U[:,None])
                
            loss.backward()
            optimizer_adam.step()

            # loss_history.append(loss.item())  # 记录loss
            
            # # 每1000步打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f'    ADAM Epoch [{epoch + 1}/{adam_epoch}], Loss: {loss.item()}')
            if loss.item() < 1e-6:
                break
            
        # LBFGS优化器    
        # optimizer_lbfgs = optim.LBFGS(self.parameters(), tolerance_grad=1e-7，tolerance_change=2e-10, line_search_fn="strong_wolfe")      
        optimizer_lbfgs = optim.LBFGS(self.parameters(), max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")    

        lbfgs_iteration = [0]  # 使用列表以便在内部函数中修改值Tolerance_grad=1E-7
        def closure():
            self.zero_grad()
            U_nn = self(X)
            if U.shape[-1] == U_nn.shape[-1] == 2:  # 向量情况
                loss = criterion(U_nn[:, 0], U[:, 0]) + criterion(U_nn[:, 1], U[:, 1])
            else:  # 标量情况e
                loss = criterion(U_nn, U[:,None])
            loss.backward()
            
            lbfgs_iteration[0] += 1
            loss_history.append(loss.item())  # 记录loss
            
            if lbfgs_iteration[0] % 100 == 0:
                print(f'    LBFGS Iteration [{lbfgs_iteration[0]}], Loss: {loss.item()}')
            return loss

        optimizer_lbfgs.step(closure)
        
        # 重新计算最终损失值
        U_nn = self(X)
        final_loss = criterion(U_nn, U[:,None]).item()
        print(f"    Training complete. Final Loss after ADAM and LBFGS optimization: {final_loss:.6e}")
        return loss_history
    
    def predict(self, X):
        """预测函数"""
        # 确保网络在评估模式
        self.eval()

        # 归一化输入
        X = self.normalize(X, self.X_min, self.X_max)
        X = torch.from_numpy(X).to(self.device).double()

        # 网络预测
        with torch.no_grad():
            U_nn = self(X).cpu().numpy()

        # 反归一化输出
        U_pred = self.denormalize(U_nn, self.U_min, self.U_max).squeeze(-1)
        return U_pred
    

class ELM(nn.Module):
    def __init__(self, mlp_layers, act='tanh', w_init='xavier_normal', b_init='normal'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
        # 参数保存
        self.mlp_layers = mlp_layers
        self.act = act
        self.w_init = w_init
        self.b_init = b_init

        # 状态初始化
        self.coef_solution = None
        self.build_network()
        
    def build_network(self):
        self.model = nn.Sequential()
        for i in range(len(self.mlp_layers) - 1):
            linear = nn.Linear(self.mlp_layers[i], self.mlp_layers[i+1], bias=True)
            apply_weight_init(linear, self.w_init)
            apply_bias_init(linear, self.b_init)
            self.model.add_module(f'fc{i+1}', linear)
            self.model.add_module(f'act{i+1}', get_activation_fn(self.act))
        self.to(self.device).double()
        
    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).to(self.device).double()
        y_train = torch.from_numpy(y_train).to(self.device).double()

        H = self.forward(X_train)
        coef_solut = torch.linalg.lstsq(H, y_train)
        self.coef_solution = coef_solut.solution

        # 计算训练损失
        y_pred = H @ self.coef_solution
        loss = torch.mean((y_train - y_pred) ** 2)
        print(f"        Training complete. Final Loss: {loss:.6e}")

    def predict(self, X):
        X = torch.from_numpy(X).to(self.device).double()
        H_eval = self.forward(X)
        return (H_eval @ self.coef_solution).detach().cpu().numpy()
    
    
class RBF(nn.Module):
    def __init__(self, mlp_layers, gamma=None):
        super().__init__()
        self.mlp_layers = mlp_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.coef_solution = None

    def build_centers(self, X_train):
        # 从训练集中随机选择部分点作为 RBF 中心
        N = X_train.shape[0]
        idx = torch.randperm(N)[:self.mlp_layers[1]]
        self.centers = X_train[idx].detach().clone()  # [L, d]

        # 自动设置 gamma
        if self.gamma is None:
            from sklearn.neighbors import NearestNeighbors
            with torch.no_grad():
                X_cpu = X_train.cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=2).fit(X_cpu)
                distances, _ = nbrs.kneighbors(X_cpu)
                avg_spacing = distances[:, 1].mean()
                self.gamma = 1.0 / (avg_spacing ** 2)
            print(f"[RBFELM] Estimated gamma: {self.gamma:.4e}")

    def forward(self, X_train):
        x = X_train.unsqueeze(1)  # [N, 1, d]
        c = self.centers.unsqueeze(0)  # [1, L, d]
        dist_sq = torch.sum((x - c)**2, dim=-1)  # [N, L]
        return torch.exp(-self.gamma * dist_sq)
    
    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).to(self.device).double()
        y_train = torch.from_numpy(y_train).to(self.device).double()

        self.build_centers(X_train)
        
        H = self.forward(X_train)
        self.coef_solution = torch.linalg.lstsq(H, y_train).solution
        y_pred = H @ self.coef_solution
        loss = torch.mean((y_train - y_pred) ** 2)
        print(f"        Training complete. Final Loss: {loss:.6e}")

    def predict(self, X):
        X = torch.from_numpy(X).to(self.device).double()
        H_eval = self.forward(X)
        return (H_eval @ self.coef_solution).detach().cpu().numpy()    
    
       
class RBFNN(nn.Module):
    def __init__(self, mlp_layers, gamma=None, act='tanh', w_init='xavier_normal', b_init='normal'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 参数保存
        self.gamma = gamma
        self.mlp_layers = mlp_layers
        self.act = act
        self.w_init = w_init
        self.b_init = b_init
        
        # 状态初始化
        self.centers = None
        self.coef_solution = None
        self.nn = nn.Sequential()
        
    def build_centers(self, X_train):
        # 从训练集中随机选择部分点作为 RBF 中心
        N = X_train.shape[0]
        idx = torch.randperm(N)[:self.mlp_layers[1]]
        self.centers = X_train[idx].detach().clone()  # [L, d]

        # 自动设置 gamma
        if self.gamma is None:
            from sklearn.neighbors import NearestNeighbors
            with torch.no_grad():
                X_cpu = X_train.cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=2).fit(X_cpu)
                distances, _ = nbrs.kneighbors(X_cpu)
                avg_spacing = distances[:, 1].mean()
                self.gamma = 1.0 / (avg_spacing ** 2)
            print(f"[RBFELM] Estimated gamma: {self.gamma:.4e}")
            
    def build_nn_layer(self):
        """封装神经网络层的构建和初始化"""
        linear = nn.Linear(self.centers.shape[0], self.mlp_layers[2], bias=True)
        apply_weight_init(linear, self.w_init)
        apply_bias_init(linear, self.b_init)
        self.nn.add_module(f'fc{1}', linear)
        self.nn.add_module(f'act{1}', get_activation_fn(self.act))
        self.nn.to(self.device).double()
        
    def forward(self, X_train):
        x = X_train.unsqueeze(1)  # [N, 1, d]
        c = self.centers.unsqueeze(0)  # [1, L, d]
        dist_sq = torch.sum((x - c)**2, dim=-1)  # [N, L]
        rbf_out = torch.exp(-self.gamma * dist_sq)       
        # NN部分
        return self.nn(rbf_out)

    def fit(self, X_train, y_train):
        X_train = torch.from_numpy(X_train).to(self.device).double()
        y_train = torch.from_numpy(y_train).to(self.device).double()

        self.build_centers(X_train)
        self.build_nn_layer()
        
        H = self.forward(X_train)
        self.coef_solution = torch.linalg.lstsq(H, y_train).solution
        y_pred = H @ self.coef_solution
        loss = torch.mean((y_train - y_pred) ** 2)
        print(f"        Training complete. Final Loss: {loss:.6e}")

    def predict(self, X):
        X = torch.from_numpy(X).to(self.device).double()
        H_eval = self.forward(X)
        return (H_eval @ self.coef_solution).detach().cpu().numpy()
    
    
def quick_clone(src, dst):
    dst.centers = src.centers.clone()
    dst.coef_solution = src.coef_solution.clone()
    dst.gamma = src.gamma
    if hasattr(src, 'nn') and hasattr(dst, 'nn'):
        dst.nn.load_state_dict(src.nn.state_dict())