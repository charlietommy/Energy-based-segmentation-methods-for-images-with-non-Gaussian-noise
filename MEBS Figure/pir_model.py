# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 22:28:06 2025

@author: charlietommy
"""

# pir_model.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

class PolarInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    极坐标交互回归 (Polar Interaction Regression, PIR)

    该模型假设一个点的输出值不仅取决于其自身特征，还受到其他点
    在距离和方向上的加权影响。

    参数:
    ----------
    K : int, default=1
        角向权重函数中傅里叶级数的阶数。
        K=1 可以捕捉一个主导方向, K=2 可以捕捉更复杂的方向模式。

    verbose : int, default=1
        是否在优化过程中打印信息。0为不打印, 1为打印最终结果, 2为打印每次迭代。
    """
    def __init__(self, K=1, verbose=1):
        self.K = K
        self.verbose = verbose
        # 初始化模型参数
        self.beta_ = None
        self.alpha_ = None
        self.sigma_r_ = None
        self.sigma_eps_ = None
        self.log_det_S_ = None # 雅可比行列式的对数
        self.n_samples_ = 0
        self.n_features_ = 0

    def _compute_polar_coords(self, coords):
        """计算所有点对之间的距离和角度矩阵"""
        # 距离矩阵 r
        r_matrix = squareform(pdist(coords))
        
        # 角度矩阵 theta
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        theta_matrix = np.arctan2(diff[:, :, 1], diff[:, :, 0])
        
        return r_matrix, theta_matrix

    def _construct_S_matrix(self, sigma_r, alpha, r, theta):
        """构建 S = I - A 矩阵"""
        # 径向权重 w(r)
        w = np.exp(-r**2 / (2 * sigma_r**2))
        np.fill_diagonal(w, 0) # 点自身对自身无影响
        
        # 角向权重 g(theta)
        g = np.ones_like(theta)
        for k in range(1, self.K + 1):
            alpha_c_k = alpha[2 * (k - 1)]
            alpha_s_k = alpha[2 * (k - 1) + 1]
            g += alpha_c_k * np.cos(k * theta) + alpha_s_k * np.sin(k * theta)
        
        A = w * g * self.y_.reshape(1, -1) # 使用 y_j 的值作为影响源
        S = np.identity(self.n_samples_) - A
        return S

    def _concentrated_log_likelihood(self, params, X, y, r, theta):
        """
        计算浓缩对数似然函数的负值 (用于最小化)
        params[0] = sigma_r
        params[1:] = alpha
        """
        sigma_r = params[0]
        alpha = params[1:]
        
        # 安全检查: 带宽不能为负
        if sigma_r <= 1e-6:
            return np.inf

        # 1. 构建 S 矩阵
        S = self._construct_S_matrix(sigma_r, alpha, r, theta)
        
        # 2. 计算雅可比行列式 |S|
        sign, log_det_S = np.linalg.slogdet(S)
        if sign <= 0: # 行列式为负或零，无效参数
            return np.inf
        
        # 3. 求解 beta_hat 和 sigma_eps_hat
        Sy = S @ y
        try:
            # 使用伪逆增加稳定性
            X_inv = np.linalg.pinv(X.T @ X)
            beta_hat = X_inv @ X.T @ Sy
            residuals = Sy - X @ beta_hat
            sigma_eps_sq_hat = (residuals.T @ residuals) / self.n_samples_
        except np.linalg.LinAlgError:
            return np.inf

        if sigma_eps_sq_hat <= 1e-9:
            return np.inf
            
        # 4. 计算浓缩对数似然
        # 我们要最大化 Lc, 等于最小化 -Lc
        neg_Lc = (self.n_samples_ / 2) * np.log(sigma_eps_sq_hat) - log_det_S
        
        if self.verbose > 1:
            print(f"Trying params: sigma_r={sigma_r:.4f}, alpha={alpha}, neg_Lc={neg_Lc:.4f}")

        return neg_Lc
    
    def fit(self, X, y, coords):
        """
        拟合PIR模型

        参数:
        ----------
        X : array-like, shape (n_samples, n_features)
            特征矩阵 (外生变量)
        y : array-like, shape (n_samples,)
            目标变量 (因变量)
        coords : array-like, shape (n_samples, 2)
            每个样本点的空间坐标 (x, y)
        """
        self.n_samples_, self.n_features_ = X.shape
        self.y_ = y # 存储 y 以便在构建A矩阵时使用
        
        # 预计算极坐标
        r, theta = self._compute_polar_coords(coords)
        
        # 设置优化器的初始值和边界
        # params = [sigma_r, alpha_1c, alpha_1s, ...]
        initial_params = np.zeros(1 + 2 * self.K)
        initial_params[0] = np.mean(pdist(coords)) / 2 # 启发式设置初始带宽
        
        # 设置边界, sigma_r > 0, alpha无边界
        bounds = [(1e-6, None)] + [(None, None)] * (2 * self.K)
        
        if self.verbose > 0:
            print("Starting PIR model optimization...")

        # 使用 scipy.optimize.minimize 进行优化
        result = minimize(
            self._concentrated_log_likelihood,
            x0=initial_params,
            args=(X, y, r, theta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': self.verbose > 1}
        )
        
        if not result.success:
            print(f"Warning: Optimization failed. Message: {result.message}")

        # 提取最优参数
        self.sigma_r_, self.alpha_ = result.x[0], result.x[1:]
        
        # 用最优参数计算最终的 beta 和 sigma_eps
        S_final = self._construct_S_matrix(self.sigma_r_, self.alpha_, r, theta)
        Sy_final = S_final @ y
        X_inv = np.linalg.pinv(X.T @ X)
        self.beta_ = X_inv @ X.T @ Sy_final
        residuals = Sy_final - X @ self.beta_
        self.sigma_eps_ = np.sqrt((residuals.T @ residuals) / self.n_samples_)
        self.log_det_S_ = np.linalg.slogdet(S_final)[1]
        
        if self.verbose > 0:
            print("PIR model fitting complete.")
            print(f"  - Optimal sigma_r: {self.sigma_r_:.4f}")
            print(f"  - Optimal alpha: {self.alpha_}")
            print(f"  - Estimated beta: {self.beta_}")
            print(f"  - Estimated sigma_eps: {self.sigma_eps_:.4f}")
            
        return self

    def predict(self, X):
        """
        进行样本内预测 (In-sample prediction)。
        
        注意：PIR是一个内生模型，真正的样本外预测非常复杂。
        这里的预测主要用于模型评估和R方计算。
        """
        if self.beta_ is None:
            raise RuntimeError("You must fit the model before predicting.")
        
        # 预测值 y_hat = (I - A)^-1 * (X*beta)
        # 这是一个简化且不完全正确的预测。
        # 理论上 y_hat = X*beta + A*y
        # 我们使用后者来进行样本内评估
        S = self._construct_S_matrix(self.sigma_r_, self.alpha_, 
                                     *self._compute_polar_coords(coords_for_X))
        A = np.identity(self.n_samples_) - S
        y_pred = X @ self.beta_ + A @ self.y_

        return y_pred
        
    def score(self, X, y, coords_for_X):
        """
        计算模型的 R-squared
        """
        # 注意: 这里需要传入 coords_for_X, 因为预测依赖于坐标
        y_pred = self.predict(X, coords_for_X)
        return r2_score(y, y_pred)