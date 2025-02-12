import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위해 seed 설정
np.random.seed(42)

# 2차원 공간, 각 행렬의 열의 개수
n = 2       # 공간 차원
m = 50      # 샘플 수

# 1. 공통 subspace(1차원) 및 각 행렬의 고유 subspace(1차원) 구성
# ------------------------------------------------------------------
# 공통 subspace: R^2 내의 임의의 단위벡터
c = np.random.randn(n, 1)
c = c / np.linalg.norm(c)

# W+의 고유 subspace: c와 직교하는 단위벡터
u_plus = np.random.randn(n, 1)
u_plus = u_plus - c * (c.T @ u_plus)   # c 성분 제거
u_plus = u_plus / np.linalg.norm(u_plus)

# W-의 고유 subspace: c와 직교하지만 u_plus와는 다른 방향의 단위벡터
u_minus = np.random.randn(n, 1)
u_minus = u_minus - c * (c.T @ u_minus)
u_minus = u_minus / np.linalg.norm(u_minus)

# 각 행렬의 열들을 생성하기 위한 계수 (랜덤)
a = np.random.randn(1, m)   # W+의 공통 성분 계수
b = np.random.randn(1, m)   # W+의 고유 성분 계수
d = np.random.randn(1, m)   # W-의 공통 성분 계수
e = np.random.randn(1, m)   # W-의 고유 성분 계수

# 행렬 구성: 각 열은 공통 성분과 고유 성분의 선형 결합으로 구성됨
W_plus = c @ a + u_plus @ b    # W+는 공통(c)와 u_plus 성분의 결합
W_minus = c @ d + u_minus @ e  # W-는 공통(c)와 u_minus 성분의 결합

# 2. SVD를 이용하여 각 행렬의 column space의 orthonormal basis 추출
# ------------------------------------------------------------------
U_plus, S_plus, Vh_plus = np.linalg.svd(W_plus, full_matrices=False)
U_minus, S_minus, Vh_minus = np.linalg.svd(W_minus, full_matrices=False)

# 3. 두 subspace의 공통 부분 추출 (두 basis의 교차 분석)
#    U_plus와 U_minus 사이의 내적행렬 M의 SVD를 이용
M = U_plus.T @ U_minus
U1, s, Vh = np.linalg.svd(M)

print("U_plus.T @ U_minus의 singular values:", s)
# singular value가 1에 가까우면 공통한 방향이 있다는 의미입니다.
# 이 경우, s[0]가 1에 가까워야 합니다.

# U_plus의 basis에서 공통 부분에 해당하는 좌표는 U1[:,0]에 해당하므로,
# 원래 공간에서의 공통 subspace 방향은 다음과 같이 구할 수 있습니다.
v_common = U_plus @ U1[:, 0].reshape(-1, 1)
v_common = v_common / np.linalg.norm(v_common)  # 단위벡터로 정규화

print("실제 공통 방향 c:", c.ravel())
print("추출한 공통 방향 v_common:", v_common.ravel())
# (부호가 반대일 수 있음에 주의)

# 4. 공통 subspace에 대한 투영행렬 구성
P_common = v_common @ v_common.T

# 5. W_minus에서 공통 부분 제거하여 고유 정보만 남기기
W_minus_unique = W_minus - P_common @ W_minus

# 6. W+에 W_minus의 고유 정보를 합치기
W_plus_new = W_plus + W_minus_unique

# 7. 시각화
# ------------------------------------------------------------------
# 2D 평면 상에 각 행렬의 열들을 점으로 표시하여 분포 비교
plt.figure(figsize=(12, 5))

# (a) 원래 W+ (파란색)와 W- (빨간색)을 표시하고, 공통 subspace (검은 점선)를 그립니다.
plt.subplot(1, 2, 1)
plt.scatter(W_plus[0, :], W_plus[1, :], label=r'$W^+$', color='blue', alpha=0.7)
plt.scatter(W_minus[0, :], W_minus[1, :], label=r'$W^-$', color='red', alpha=0.7)
plt.title('Original $W^+$ and $W^-$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')

# 공통 subspace (원점을 지나고 v_common 방향으로의 직선) 그리기
line_x = np.linspace(-3, 3, 100)
# v_common = [v0, v1]^T 이므로 y = (v1/v0)*x
line_y = (v_common[1, 0] / v_common[0, 0]) * line_x
plt.plot(line_x, line_y, 'k--', label='Common subspace')

# (b) 고유정보를 합친 W_plus_new (녹색)와 원래의 W+ (테두리만 파란색) 비교
plt.subplot(1, 2, 2)
plt.scatter(W_plus_new[0, :], W_plus_new[1, :], label=r'$W^+_{\rm new}$', color='green', alpha=0.7)
plt.scatter(W_plus[0, :], W_plus[1, :], label=r'$W^+$ (original)', 
            facecolors='none', edgecolors='blue', s=80)
plt.title(r'$W^+$ with $W^-$ unique part added')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.plot(line_x, line_y, 'k--', label='Common subspace')

plt.tight_layout()
plt.show()
