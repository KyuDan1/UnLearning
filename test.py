import numpy as np
from scipy.linalg import svd

class LoRAUnlearner:
    def __init__(self, rank_common=8, beta=0.5, use_fisher=False):
        """
        LoRA 언러닝을 위한 최적화 클래스
        
        Parameters:
        rank_common (int): 공통 정보 랭크 차원 (기본값: 8)
        beta (float): 언러닝 강도 계수 (0~1, 기본값: 0.5)
        use_fisher (bool): Fisher 정보 기반 조정 사용 여부 (기본값: False)
        """
        self.rank_common = rank_common
        self.beta = beta
        self.use_fisher = use_fisher
        self.U_common = None
        self.F = None

    def _compute_common_subspace(self, W_plus, W_minus):
        """공통 subspace 계산을 위한 다중 SVD 계층화 연산"""
        # 각 가중치 행렬 SVD 분해
        U_plus, _, _ = svd(W_plus, full_matrices=False)
        U_minus, _, _ = svd(W_minus, full_matrices=False)
        
        # 공통 기저 추출을 위한 결합 SVD
        joint_U = np.hstack((U_plus[:, :self.rank_common], 
                           U_minus[:, :self.rank_common]))
        U_joint, _, _ = svd(joint_U, full_matrices=False)
        
        self.U_common = U_joint[:, :self.rank_common]
        
    def _compute_fisher_matrix(self, W):
        """Fisher 정보 행렬 계산 (대각 근사 버전)"""
        self.F = np.diag(1.0 / (np.std(W, axis=1)**2 + 1e-8))

    def fit(self, W_plus, W_minus):
        """
        공통 성분 및 Fisher 정보 학습
        
        Parameters:
        W_plus (np.ndarray): 양성 데이터로 파인튜닝된 LoRA 가중치
        W_minus (np.ndarray): 불량 데이터로 파인튜닝된 LoRA 가중치
        """
        # 1. 공통 부공간 계산
        self._compute_common_subspace(W_plus, W_minus)
        
        # 2. Fisher 정보 행렬 계산
        if self.use_fisher:
            self._compute_fisher_matrix(np.hstack((W_plus, W_minus)))

    def unlearn(self, W_plus):
        """
        언러닝 수행 메인 함수
        
        Parameters:
        W_plus (np.ndarray): 원본 LoRA 가중치
        
        Returns:
        np.ndarray: 언러닝된 가중치
        """
        # 공통 성분 투영 계산
        common_component = self.U_common @ (self.U_common.T @ W_plus)
        
        # 순수 불량 성분 추출
        W_double_minus = W_plus - common_component
        
        # Fisher 정보 기반 조정
        if self.use_fisher:
            W_double_minus = self.F @ W_double_minus
        
        # 안정화 언러닝 연산
        unlearned_W = W_plus - self.beta * np.sign(W_plus) * np.abs(W_double_minus)
        
        return unlearned_W
    
# 사용 예시
if __name__ == "__main__":
    # 예제 가중치 생성 (d=768, r=16)
    d, r = 768, 16
    W_plus = np.random.randn(d, r)
    W_minus = np.random.randn(d, r)
    
    # 언러닝 수행
    unlearner = LoRAUnlearner(rank_common=8, beta=0.7, use_fisher=True)
    unlearner.fit(W_plus, W_minus)
    unlearned_W = unlearner.unlearn(W_plus)
    
    print(f"Unlearned 가중치 차원: {unlearned_W.shape}")