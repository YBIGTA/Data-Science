## Eigendecomposition

### Diagonalization: $D=V^{-1}AV$  ($\equiv A=VDV^-1$)
#### Notation
$A\in R^{n\times n}$: 정사각행렬

$D\in R^{n\times n}$: 대각행렬 (대각성분을 제외한 모든 값이 0인 행렬)

ex) $\begin{bmatrix} 3 & 0 \\ 0 & -1  \end{bmatrix} $

$V\in R^{n\times n}$: invertible 행렬 (역행렬이 있음)

#### V는 무엇? D는 무엇?

식을 바꿔보자..

$D=V^{-1}AV$

==> $VD=AV$

1. $VD=\begin{bmatrix} \mathbf{v_1} & \mathbf{v_2} & \cdots & \mathbf{v_n} \end{bmatrix}
\begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\cdots & \cdots & \cdots & 0 \\
0 & \cdots & 0 & \lambda_n \\
\end{bmatrix}
= \begin{bmatrix} \lambda_1 \mathbf{v_1} & \lambda_2 \mathbf{v_2} & \cdots & \lambda_n \mathbf{v_n} \end{bmatrix}
$
2. $AV=A\begin{bmatrix} \mathbf{v_1} & \mathbf{v_2} & \cdots & \mathbf{v_n} \end{bmatrix}=
\begin{bmatrix} \mathbf{Av_1} & \mathbf{Av_2} & \cdots & \mathbf{Av_n} \end{bmatrix}$


$\therefore \begin{bmatrix} \lambda_1 \mathbf{v_1} & \lambda_2 \mathbf{v_2} & \cdots & \lambda_n \mathbf{v_n} \end{bmatrix}=
\begin{bmatrix} \mathbf{Av_1} & \mathbf{Av_2} & \cdots & \mathbf{Av_n} \end{bmatrix}$

### $\therefore$ $A\mathbf{v_i} = \lambda_i \mathbf{v_i}, \forall i\in [1,n]$

따라서 $v_i$ 는 $A$ 의 eigen vector 들이 되고 $\lambda_i$ 는 $v_i$ 에 대응하는 eigen value 가 된다.

다시 정리해서 말하면
$A=VDV^{-1}$ 에서
1. $V$ 는 $A$의 eigen vector $v_i$ 들을 column vector 로 가지는 행렬이고

2. $D$ 는 $V$ 의 각 column vector $v_i$ 에 대응하는 eigen value 들을 대각성분으로 가지는 diagonal matrix 이다!

### Note that
$V^{-1}$ 가 존재하기 위해서는 $A$ 의 eigen vector $v_i$ 들 ($V$ 의 column vectors)이 서로 linearly independent 해야 한다. 

### 마지막으로 정리해서 말하자면
* $A=VDV^{-1}$ 로 square matrix $A$를 decomposition 하는 것을 eigendecomposition 이라고 하며 

* eigen 이라는 이름에서 알 수 있듯이
$V$ 는 $A$의 eigen vector 들을 column vector 로 가지는 행렬이고 $D$는 corresponding eigen value 들을 대각성분으로 가지는 diagonal matrix 이다

* 이 때 $V^{-1}$ 가 존재하기 위해서는 $A$의 eigen vector 들이 서로 linearly independent 해야하므로 eigendecomposition 이 언제나 가능한 것은 아니다. 

* 참고: 특수한 경우로 $A$가 symmetric matrix 라면 언제나 eigendecomposition 이 가능하다. covariance matrix 가 symmatric matrix 이므로 covariance matrix 를 분석 할 때 eigendecomposition 이 많이 사용된다.


### $A=VDV^{-1}$ 의 의미

어떤 벡터 $x$ 에 행렬 $A$를 곱함으로써 linear transformation 을 한다고 해보자

$T(x)=Ax=VDV^{-1}x=V(D(V^{-1}x))$

#### 1. $V^{-1}x$ 의 의미: Change of basis
* $V^{-1}x=y$ 라고 해보면
* $x=Vy$ 이고 다르게 쓰면 ($V$가 $2\times 2$ 행렬이라고 가정)
* $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} x=
\begin{bmatrix} \mathbf{v_1} & \mathbf{v_2} \end{bmatrix}
y$
* 즉, 좌변과 우변의 벡터는 서로 동일한 벡터인데 좌변의 경우 standard basis 와 $x$라는 coefficient 로 벡터를 표현한 것이고 우변의 경우 $\mathbf{v_1},\mathbf{v_2}$ 라는 두 basis 와 $y$라는 coefficient 로 벡터를 표현한 것이다. ($\mathbf{v_i},\forall i\in [1,n]$ 은 서로 linearly independent 이어야 하므로 $R^n$ space 를 span 할 수 있다. 즉 어떤 벡터 $x$든 표현할 수 있다.)
* 다시 말해서 $y$는 $x$라는 벡터를 $\{v_1, v_2\}$ 의 linear combination 으로 표현했을 때의 coefficient 가 된다.


#### 2. $Dy$ 의 의미: Element-wise Scaling
* $z=Dy$ 라고 할 때 $D$는 대각행렬이고 대각행렬의 linear transformation 연산은 매우 간단하다.
* ex) 
$D=\begin{bmatrix} -1 & 0 \\ 0 & 2\end{bmatrix}$,
$y=\begin{bmatrix} 2 \\ 1\end{bmatrix}$
$Dy=
\begin{bmatrix} -1 & 0 \\ 0 & 2\end{bmatrix}
\begin{bmatrix} 2 \\ 1\end{bmatrix}
= \begin{bmatrix} (-1)\times 2 \\ 2\times 1\end{bmatrix}
= \begin{bmatrix} -2 \\ 2 \end{bmatrix}
$
* 위의 예에서 $y$는 $\{{v_1, v_2}\}$를 basis 로 하는 공간의 coefficient 를 의미했었다. 그런데 $z_i=\lambda_i y_i$ 이고 따라서 $z_i$ 에는 $y_i$ 를 제외한 어떤 다른 $y_j(j\neq i)$도 관여하지 않으므로 $Dy$ 의 연산은 axis (각 basis) 를 따라 Scaling 해주는 연산을 의미한다.

#### 3. $Vz$ 의 의미: Back to Original Basis
* $z$는 $\{v_1,v_2\}$를 basis로 표현한 coefficient 이다.
* $Vz=w$ 라고 했을 때 $Vz=I w$로 표현할 수 있으므로 #1 에서와 동일한 이유로 $w$는 $\{v_1,v_2\}$를 basis 로 했을 때의 coefficient 인 $z$ 를 standard basis 의 coefficient 로 바꾸는 과정을 의미한다.

#### $A$ 라는 linear transformation 을 여러 번 반복한다면?
$A=VDV^{-1}$ 일 때, $A^{k}=(VDV^{-1})(VDV^{-1})\cdots(VDV^{-1})=VD^kV^{-1}$

$A\times A\times \cdots \times Ax$ 보다 훨씬 효율적이다.

(참고: 위의 선형대수 내용이 RNN gradient vanishing 문제에 적용된 사례 - "orthogonal initialization" https://smerity.com/articles/2016/orthogonal_init.html)