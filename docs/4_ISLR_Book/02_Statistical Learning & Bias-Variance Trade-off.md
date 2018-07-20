# 02 Statistical Learning

------------------------------------------------------------------------

#### Yeon Soo, Choi

------------------------------------------------------------------------

![logo](./logo.png)
===================

What is Statistical Learning?
-----------------------------

------------------------------------------------------------------------

**통계적 학습**이란, **입력값** (Input Variables, ![X](https://latex.codecogs.com/png.latex?X "X") ) 과 **출력값** (Output Variables, ![Y](https://latex.codecogs.com/png.latex?Y "Y") ) 이 주어졌을 때 다음과 같이

![Y = f(X)+\\epsilon](https://latex.codecogs.com/png.latex?Y%20%3D%20f%28X%29%2B%5Cepsilon "Y = f(X)+\epsilon")

일반적인 형태로 입력값과 출력값의 관계를 설명할 수 있는 ![fixed](https://latex.codecogs.com/png.latex?fixed "fixed") ![unknown~function](https://latex.codecogs.com/png.latex?unknown~function "unknown~function") ![f](https://latex.codecogs.com/png.latex?f "f") 와 ![error](https://latex.codecogs.com/png.latex?error "error") ![term](https://latex.codecogs.com/png.latex?term "term") 이 존재한다고 하면 이러한 ![f](https://latex.codecogs.com/png.latex?f "f")가 ![X](https://latex.codecogs.com/png.latex?X "X")가 ![Y](https://latex.codecogs.com/png.latex?Y "Y")에 대해 제공하는 systematic information을 대표한다고 책에서는 말한다. (여기서 말하는 systematic information이란 X와 Y의 실제적 인과관계가 아닌 주어진 정보 하에 function의 형태에서 얻어질 수 있는 두 관계에 대한 정보를 의미하는 것 같다.)

결론적으로, 통계적 학습은 이러한 ![f](https://latex.codecogs.com/png.latex?f "f") 를 추정하는 것을 의미하고 ISL 교재 내에서는 이러한 ![f](https://latex.codecogs.com/png.latex?f "f") 를 추정할 수 있는 다양한 방법론 그리고 이를 평가하는 방법들에 대해서 소개하고 있다.

------------------------------------------------------------------------

### Why Estimate ![f](https://latex.codecogs.com/png.latex?f "f") ?

------------------------------------------------------------------------

![f](https://latex.codecogs.com/png.latex?f "f") 를 추정하고자 하는 데에는 **prediction, inference**로 두 가지 이유가 있다.

------------------------------------------------------------------------

#### Prediction (예측)

------------------------------------------------------------------------

우리는 알지 못하는 ![Y](https://latex.codecogs.com/png.latex?Y "Y") 에 대해 다음과 같이 ![X](https://latex.codecogs.com/png.latex?X "X") 로 예측을 할 수 있다. (error term이 averages to zero 라는 표현을 썼는데 ![\\epsilon\\sim~N(0,\\sigma^2)](https://latex.codecogs.com/png.latex?%5Cepsilon%5Csim~N%280%2C%5Csigma%5E2%29 "\epsilon\sim~N(0,\sigma^2)") 를 말하는 듯 함)

![\\hat{Y}=\\widehat{f}(X)](https://latex.codecogs.com/png.latex?%5Chat%7BY%7D%3D%5Cwidehat%7Bf%7D%28X%29 "\hat{Y}=\widehat{f}(X)")

예측 관점에서는 ![Y](https://latex.codecogs.com/png.latex?Y "Y") 에 대한 추정치 ![\\hat{f}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D "\hat{f}") 의 정확한 형태(formula)에는 관심이 없고 그저 ![Y](https://latex.codecogs.com/png.latex?Y "Y")에 대한 예측의 정확성에 관심이 있기 때문에 ![\\hat{f}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D "\hat{f}")를 종종 ![black](https://latex.codecogs.com/png.latex?black "black") ![box](https://latex.codecogs.com/png.latex?box "box") 로 취급한다.

여기서 말하는 **정확도(![Accuracy](https://latex.codecogs.com/png.latex?Accuracy "Accuracy"))** 이란?

*Minimizing the Reducible Error!*

Regression Problem 에서 흔히 쓰이는 정확도(오차)에 대한 척도 **MSE**(*Mean Squared Error*)를 살펴보면

![E(Y-\\hat{Y})^2=E\[f(X)+\\epsilon-\\hat{f(X)}\]^2=\[f(X)-\\hat{f(X)}\]^2+E\[\\epsilon^2\]=](https://latex.codecogs.com/png.latex?E%28Y-%5Chat%7BY%7D%29%5E2%3DE%5Bf%28X%29%2B%5Cepsilon-%5Chat%7Bf%28X%29%7D%5D%5E2%3D%5Bf%28X%29-%5Chat%7Bf%28X%29%7D%5D%5E2%2BE%5B%5Cepsilon%5E2%5D%3D "E(Y-\hat{Y})^2=E[f(X)+\epsilon-\hat{f(X)}]^2=[f(X)-\hat{f(X)}]^2+E[\epsilon^2]=")

![\[f(X)-\\hat{f(X)}\]^2+Var(\\epsilon)](https://latex.codecogs.com/png.latex?%5Bf%28X%29-%5Chat%7Bf%28X%29%7D%5D%5E2%2BVar%28%5Cepsilon%29 "[f(X)-\hat{f(X)}]^2+Var(\epsilon)")

![Var(\\epsilon)=E\[(\\epsilon-E\[\\epsilon\])^2\]=E\[\\epsilon^2\]-0^2=E\[\\epsilon^2\]](https://latex.codecogs.com/png.latex?Var%28%5Cepsilon%29%3DE%5B%28%5Cepsilon-E%5B%5Cepsilon%5D%29%5E2%5D%3DE%5B%5Cepsilon%5E2%5D-0%5E2%3DE%5B%5Cepsilon%5E2%5D "Var(\epsilon)=E[(\epsilon-E[\epsilon])^2]=E[\epsilon^2]-0^2=E[\epsilon^2]")

다음과 같은 두 error에 대한 정의와 위 식을 살펴보면 Reducible Error는 우리가 보다 정확한 추정치를 찾는다면 줄일 수 있고 우리가 줄일 수 없는 Irreducible Error는 그 분산에 의존할 수 밖에 없기에 직관적으로도 이해 가능하다.

**1. Reducible error** - 적절한 통계적 학습 기술을 이용해 줄일 수 있는 ![error](https://latex.codecogs.com/png.latex?error "error")

**2. Irreducible error** - 일종의 통계학에서 말하는 ![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon") (![Y](https://latex.codecogs.com/png.latex?Y "Y")를 설명가능한 변수의 부재, ![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon") 이 측정 불가능한 변동성을 포함..etc)

(Why **MSE?**) 전체적인 오차에 대한 척도를 나타낼 때 그냥 평균을 내버리면 전체적인 오차가 얼마나 큰지 가늠할 수가 없다. 평균을 구할 때 합을 구해야하는데 오차가 아무리 커도 양수, 음수가 섞여 있으면 합이 0이 돼버리는 경우가 있기 때문. 따라서 부호를 없애기 위해 제곱을 선택한다... 때때로 MSE는 값이 너무 커서 계산 속도의 효율성을 위해 제곱근을 씌운 **RMSE(Root Mean Squared Error)** 를 사용하는 것이다.

------------------------------------------------------------------------

#### Inference (추론)

------------------------------------------------------------------------

위의 경우와 다르게 때때로 우리는 ![Y](https://latex.codecogs.com/png.latex?Y "Y") 가 ![X](https://latex.codecogs.com/png.latex?X "X") 에 의해 어떻게 영향을 받는지에 대한 **해석**이 주된 관심이 될 수 있다.

이 경우 우리는 예측의 정확성보다는 ![X](https://latex.codecogs.com/png.latex?X "X")와 ![Y](https://latex.codecogs.com/png.latex?Y "Y") 의 관계에 대해 관심이 있기 때문에 이 경우 우리는 ![\\hat{f}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D "\hat{f}") 의 정확한 form 에 대해 알아야 한다. 이를테면

-   어떤 독립변수들이 종속변수와 관련이 있는가?
-   ![X\_i](https://latex.codecogs.com/png.latex?X_i "X_i")가 한 단위 변화할때 ![Y](https://latex.codecogs.com/png.latex?Y "Y")가 얼마나 증가(혹은 감소)하는가?
-   ![X](https://latex.codecogs.com/png.latex?X "X")와 ![Y](https://latex.codecogs.com/png.latex?Y "Y")의 관계가 선형 방정식의 형태로 설명될 수 있는가? 아니면 더 복잡한 형태의 관계를 가지는가?

와 같은 문제들 말이다. 어떤 문제들의 경우 이러한 **추론**과 **예측**의 관점 두 가지가 모두 필요한 경우들도 있을 것이다.

------------------------------------------------------------------------

### Trade-Off Between Prediction Accuracy and Model Interpretability

------------------------------------------------------------------------

거의 대부분의 통계적 학습 모형들은 모수적 방법(![parametric](https://latex.codecogs.com/png.latex?parametric "parametric")) 과 비모수적 방법(![non-parametric](https://latex.codecogs.com/png.latex?non-parametric "non-parametric")) 으로 나타낼 수 있다.

-   **Parametric Methods**

1.  모형 가정 (ex. linear model)
2.  모형 추정 (model fit or train)

-   장점 : 모형 추정의 간편성
-   단점 : ![f](https://latex.codecogs.com/png.latex?f "f") 의 True form과 맞지 않을 수 있다. flexible한 모형을 선택한다해도 overfitting 의 문제가 있을 수 있음

-   **Non-parametric Methods**

1.  모형 가정이 없음
2.  단지 관측값에 가장 가까운 ![\\hat{f}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D "\hat{f}") 를 찾을 뿐

-   장점 : 모형 가정이 없어 추정치의 가능한 형태가 보다 자유롭다. (![f](https://latex.codecogs.com/png.latex?f "f") 의 True form과 보다 가까워질 수 있다.)
-   단점 : 매우 많은 관측치가 필요하다. overfitting의 문제

------------------------------------------------------------------------

#### Trade-Off Between Prediction Accuracy and Model Interpretability

------------------------------------------------------------------------

![models](./models.PNG)
=======================

모형에 따른 **flexibility** 와 **interpretability** 의 **trade-off** 관계를 나타낸 그림은 위와 같다. 이를 통해 모형이 유연할수록 해석의 가능성은 떨어진다는 것을 확인할 수 있다.

추론이 주된 ![focus](https://latex.codecogs.com/png.latex?focus "focus") 라면 모형의 해석을 위해 restrictive method 를 선호할 것이다.

------------------------------------------------------------------------

### Supervised Learning vs. Unsupervised Learning

------------------------------------------------------------------------

**Supervised Learning** 과 종속변수가 주어져있지 않는 **Unsupervised Learning** 에 대한 간단한 설명 그리고 ![n](https://latex.codecogs.com/png.latex?n "n")개 그리고 ![m](https://latex.codecogs.com/png.latex?m "m") (![m](https://latex.codecogs.com/png.latex?m "m") &lt; ![n](https://latex.codecogs.com/png.latex?n "n"))개의 관측치가 있다고 할 때 나머지 ![n](https://latex.codecogs.com/png.latex?n "n")-![m](https://latex.codecogs.com/png.latex?m "m") 개의 관측치가 종속변수가 없다고 할 때의 경우인 **Semi-supervised Learning** 에 대해서도 책은 있다 정도만 소개하고 있다.

------------------------------------------------------------------------

### Regression vs. Classification

------------------------------------------------------------------------

종속변수가 **양적변수** ( Quantitative Variable ) 일 때의 **Regression Problem**, **질적변수** ( Quantitative Variable ) 일 때의 **Classification Problem** 에 대해 책에서는 정의하고 있다.

------------------------------------------------------------------------

Assessing Model Accuracy
------------------------

------------------------------------------------------------------------

좋은 결과를 위해서는 올바른 모형의 선택이 반드시 뒤따라야 할 것이다. 이를 위해 모형의 평가에 대한 이야기를 책에서는 다루고 있다.

------------------------------------------------------------------------

### Measuring the Quality of Fit

------------------------------------------------------------------------

모형의 성능을 평가하기 위해 예측치와 관측치가 잘 맞는지 확인해야 할 것이다.

![MSE=\\frac1n\\Sigma(y\_i-\\hat{f}(x\_i))^2 ](https://latex.codecogs.com/png.latex?MSE%3D%5Cfrac1n%5CSigma%28y_i-%5Chat%7Bf%7D%28x_i%29%29%5E2%20 "MSE=\frac1n\Sigma(y_i-\hat{f}(x_i))^2 ")

회귀 문제에서 가장 많이 사용하는 loss function인 **MSE**를 예로 들어 책에서는 모형의 성능을 평가하기 위해서는 **Training MSE** 보다 아직 관찰되지 않은 관측치에 대한 예측값의 **Test MSE**를 최소화하는 것이 중요하다고 설명한다.

대다수의 통계적 모형들은 주어진 Training set 하에서의 MSE를 최소화하는 계수를 추정하기 때문이다.

이에 대한 해결책으로의 **cross-validation** 과 같은 방법들은 이후에 다룬다.

------------------------------------------------------------------------

\[실습\] Bias - Variance Trade - Off
------------------------------------

------------------------------------------------------------------------

#### 고집이 센 사람 vs. 귀가 얇은 사람

지도학습 알고리즘에서 과다적합 상황에서는 편향이 낮고 분산은 높다고 하고, 과소적합 상황에서는 편향이 높고 분산이 낮다고 한다.

이에 대해 보다 깊은 이해를 위해 이론을 복습하고 직접 실습을 통해 몸으로 느껴보고 싶었다.

![y=f(x)+\\epsilon](https://latex.codecogs.com/png.latex?y%3Df%28x%29%2B%5Cepsilon "y=f(x)+\epsilon"), ![\\epsilon~~\\sim~~ iid~N(0,\\sigma^2)](https://latex.codecogs.com/png.latex?%5Cepsilon~~%5Csim~~%20iid~N%280%2C%5Csigma%5E2%29 "\epsilon~~\sim~~ iid~N(0,\sigma^2)"),

![\\hat{y}=\\hat{f}(x)](https://latex.codecogs.com/png.latex?%5Chat%7By%7D%3D%5Chat%7Bf%7D%28x%29 "\hat{y}=\hat{f}(x)")

이라고 할 때,

새로운 관측치 (![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0"),![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0")) 에 대한 **expected test MSE**은 다음과 같이 나타낼 수 있다.

![E\[(y\_0-\\hat{f}(x\_0))^2\]](https://latex.codecogs.com/png.latex?E%5B%28y_0-%5Chat%7Bf%7D%28x_0%29%29%5E2%5D "E[(y_0-\hat{f}(x_0))^2]")

먼저 **Bias**와 **Variance**에 대해 설명하면,

-   **추정치의 Bias **

true function ![f](https://latex.codecogs.com/png.latex?f "f") 의 ![f(x\_0)](https://latex.codecogs.com/png.latex?f%28x_0%29 "f(x_0)") 과 예측값 ![\\hat{f}(x\_0)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x_0%29 "\hat{f}(x_0)")의 **기댓값**의 차

왜 기댓값의 차일까?

이는 우리의 모델 ![\\hat{f}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D "\hat{f}")가 **추정치**이며 ![random~variable](https://latex.codecogs.com/png.latex?random~variable "random~variable") 이라는 사실에서 기인한다.

![f(x\_0)](https://latex.codecogs.com/png.latex?f%28x_0%29 "f(x_0)")에서 true function ![f](https://latex.codecogs.com/png.latex?f "f") 는 fixed unknown function 이므로 ![f(x\_0)](https://latex.codecogs.com/png.latex?f%28x_0%29 "f(x_0)") 또한 상수이다. 따라서 ![E\[f(x\_0)\]=f(x\_0)](https://latex.codecogs.com/png.latex?E%5Bf%28x_0%29%5D%3Df%28x_0%29 "E[f(x_0)]=f(x_0)") , 이며 ![Var(f(x\_0))=0](https://latex.codecogs.com/png.latex?Var%28f%28x_0%29%29%3D0 "Var(f(x_0))=0") 이다.

따라서 **Bias**는 다음과 같이 수식으로 나타낼 수 있을 것이고

![Bias\[\\hat{f}(x\_0)\]=E\[\\hat{f}(x\_0)\]-f(x\_0)=E\[\\hat{f}(x\_0)-f(x\_0)\]](https://latex.codecogs.com/png.latex?Bias%5B%5Chat%7Bf%7D%28x_0%29%5D%3DE%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%3DE%5B%5Chat%7Bf%7D%28x_0%29-f%28x_0%29%5D "Bias[\hat{f}(x_0)]=E[\hat{f}(x_0)]-f(x_0)=E[\hat{f}(x_0)-f(x_0)]")

**Bias의 제곱**은 다음과 같을 것이다.

![Bias^2\[\\hat{f}(x\_0)\]=E\[\\hat{f}(x\_0)\]^2+f(x\_0)^2-2E\[\\hat{f}(x\_0)\]f(x\_0)](https://latex.codecogs.com/png.latex?Bias%5E2%5B%5Chat%7Bf%7D%28x_0%29%5D%3DE%5B%5Chat%7Bf%7D%28x_0%29%5D%5E2%2Bf%28x_0%29%5E2-2E%5B%5Chat%7Bf%7D%28x_0%29%5Df%28x_0%29 "Bias^2[\hat{f}(x_0)]=E[\hat{f}(x_0)]^2+f(x_0)^2-2E[\hat{f}(x_0)]f(x_0)")

-   **추정치의 Variance**

예측치가 평균적으로 그 평균에서 얼마만큼의 폭으로 변동하느냐에 대한 의미

분산이 크다는 것은 예측치의 범위가 넓다는 것이고, 이런 모델에서 이상치가 예측의 범위에 포함되어있을 수 있었으므로 **(Noise까지 Signal로 인식했다는 표현이 더 적절할 것 같다)** 과도적합이 되었을 수 있음을 의미한다.

따라서 반대로 과소적합의 경우에 분산이 작다는 것은 예측치의 범위가 좁다는 것이고, 이는 실제 맞는 데이터도 예측의 범위에 포함이 안되어 예측이 불가능하다는 것을 의미한다. **(Signal 까지 Noise로 취급)**

따라서 **Variance**는 다음과 같이 수식으로 나타낼 수 있을 것이다.

![Var(\\hat{f}(x\_0))=E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])^2\]](https://latex.codecogs.com/png.latex?Var%28%5Chat%7Bf%7D%28x_0%29%29%3DE%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%5E2%5D "Var(\hat{f}(x_0))=E[(\hat{f}(x_0)-E[\hat{f}(x_0)])^2]")

-   **expected Test MSE**

![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0") , ![f(x\_0)](https://latex.codecogs.com/png.latex?f%28x_0%29 "f(x_0)")는 실재하는 값이고, ![E\[\\epsilon\]=0,~E\[y\_0\]=f(x\_0),~Var(y\_0)=\\sigma^2](https://latex.codecogs.com/png.latex?E%5B%5Cepsilon%5D%3D0%2C~E%5By_0%5D%3Df%28x_0%29%2C~Var%28y_0%29%3D%5Csigma%5E2 "E[\epsilon]=0,~E[y_0]=f(x_0),~Var(y_0)=\sigma^2") 이며 추정치인 ![\\hat{f}(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x%29 "\hat{f}(x)") 는 ![random~variable](https://latex.codecogs.com/png.latex?random~variable "random~variable") 이기 때문에,

![Test \\ MSE=E\[(y\_0-\\hat{f}(x\_0))^2\]=E\[(y\_0-f(x\_0)+f(x\_0)-\\hat{f}(x\_0))^2\]=](https://latex.codecogs.com/png.latex?Test%20%5C%20MSE%3DE%5B%28y_0-%5Chat%7Bf%7D%28x_0%29%29%5E2%5D%3DE%5B%28y_0-f%28x_0%29%2Bf%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5E2%5D%3D "Test \ MSE=E[(y_0-\hat{f}(x_0))^2]=E[(y_0-f(x_0)+f(x_0)-\hat{f}(x_0))^2]=") ![E\[(y\_0-f(x\_0))^2+(f(x\_0)-\\hat{f}(x\_0))^2+2(y\_0-f(x\_0))(f(x\_0)-\\hat{f}(x\_0))\]=](https://latex.codecogs.com/png.latex?E%5B%28y_0-f%28x_0%29%29%5E2%2B%28f%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5E2%2B2%28y_0-f%28x_0%29%29%28f%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5D%3D "E[(y_0-f(x_0))^2+(f(x_0)-\hat{f}(x_0))^2+2(y_0-f(x_0))(f(x_0)-\hat{f}(x_0))]=")

![Var(y\_0)+E\[(\\hat{f}(x\_0)-f(x\_0))^2\]+2E\[(y\_0-f(x\_0))(f(x\_0)-\\hat{f}(x\_0))\]](https://latex.codecogs.com/png.latex?Var%28y_0%29%2BE%5B%28%5Chat%7Bf%7D%28x_0%29-f%28x_0%29%29%5E2%5D%2B2E%5B%28y_0-f%28x_0%29%29%28f%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5D "Var(y_0)+E[(\hat{f}(x_0)-f(x_0))^2]+2E[(y_0-f(x_0))(f(x_0)-\hat{f}(x_0))]")

이다.

여기서

![E\[(\\hat{f}(x\_0)-f(x\_0))^2\]=E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\]+E\[\\hat{f}(x\_0)\]-f(x\_0))^2\]=](https://latex.codecogs.com/png.latex?E%5B%28%5Chat%7Bf%7D%28x_0%29-f%28x_0%29%29%5E2%5D%3DE%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%2BE%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5E2%5D%3D "E[(\hat{f}(x_0)-f(x_0))^2]=E[(\hat{f}(x_0)-E[\hat{f}(x_0)]+E[\hat{f}(x_0)]-f(x_0))^2]=")

![E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])^2\]+(E\[\\hat{f}(x\_0)\]-f(x\_0))^2 +](https://latex.codecogs.com/png.latex?E%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%5E2%5D%2B%28E%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5E2%20%2B "E[(\hat{f}(x_0)-E[\hat{f}(x_0)])^2]+(E[\hat{f}(x_0)]-f(x_0))^2 +")

![2E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])(E\[\\hat{f}(x\_0)\]-f(x\_0))\]=](https://latex.codecogs.com/png.latex?2E%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%28E%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5D%3D "2E[(\hat{f}(x_0)-E[\hat{f}(x_0)])(E[\hat{f}(x_0)]-f(x_0))]=")

![Var(\\hat{f}(x\_0))+Bias^2\[\\hat{f}(x\_0)\]+2E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])(E\[\\hat{f}(x\_0)\]-f(x\_0))\]](https://latex.codecogs.com/png.latex?Var%28%5Chat%7Bf%7D%28x_0%29%29%2BBias%5E2%5B%5Chat%7Bf%7D%28x_0%29%5D%2B2E%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%28E%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5D "Var(\hat{f}(x_0))+Bias^2[\hat{f}(x_0)]+2E[(\hat{f}(x_0)-E[\hat{f}(x_0)])(E[\hat{f}(x_0)]-f(x_0))]")

이고

아래의 Cross-Product Term들은 그 결과가 0이므로

-   ![E\[(y\_0-f(x\_0))(f(x\_0)-\\hat{f}(x\_0))\]](https://latex.codecogs.com/png.latex?E%5B%28y_0-f%28x_0%29%29%28f%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5D "E[(y_0-f(x_0))(f(x_0)-\hat{f}(x_0))]")

![E\[(y\_0-f(x\_0))(f(x\_0)-\\hat{f}(x\_0))\]=E\[y\_0f(x\_0)-f(x\_0)^2-y\_0\\hat{f}(x\_0)+f(x\_0)\\hat{f}(x\_0)\]=](https://latex.codecogs.com/png.latex?E%5B%28y_0-f%28x_0%29%29%28f%28x_0%29-%5Chat%7Bf%7D%28x_0%29%29%5D%3DE%5By_0f%28x_0%29-f%28x_0%29%5E2-y_0%5Chat%7Bf%7D%28x_0%29%2Bf%28x_0%29%5Chat%7Bf%7D%28x_0%29%5D%3D "E[(y_0-f(x_0))(f(x_0)-\hat{f}(x_0))]=E[y_0f(x_0)-f(x_0)^2-y_0\hat{f}(x_0)+f(x_0)\hat{f}(x_0)]=")

![f(x\_0)^2-f(x\_0)^2-E\[y\_0\\hat{f}(x\_0)\]+f(x\_0)E\[\\hat{f}(x\_0)\]=](https://latex.codecogs.com/png.latex?f%28x_0%29%5E2-f%28x_0%29%5E2-E%5By_0%5Chat%7Bf%7D%28x_0%29%5D%2Bf%28x_0%29E%5B%5Chat%7Bf%7D%28x_0%29%5D%3D "f(x_0)^2-f(x_0)^2-E[y_0\hat{f}(x_0)]+f(x_0)E[\hat{f}(x_0)]=")

![-E\[(f(x\_0)+\\epsilon)\\hat{f}(x\_0)\]+f(x\_0)E\[\\hat{f}(x\_0)\]=](https://latex.codecogs.com/png.latex?-E%5B%28f%28x_0%29%2B%5Cepsilon%29%5Chat%7Bf%7D%28x_0%29%5D%2Bf%28x_0%29E%5B%5Chat%7Bf%7D%28x_0%29%5D%3D "-E[(f(x_0)+\epsilon)\hat{f}(x_0)]+f(x_0)E[\hat{f}(x_0)]=")

![-E\[f(x\_0)\\hat{f}(x\_0)\]-E\[\\epsilon\\hat{f}(x\_0)\]+f(x\_0)E\[\\hat{f}(x\_0)\]=0](https://latex.codecogs.com/png.latex?-E%5Bf%28x_0%29%5Chat%7Bf%7D%28x_0%29%5D-E%5B%5Cepsilon%5Chat%7Bf%7D%28x_0%29%5D%2Bf%28x_0%29E%5B%5Chat%7Bf%7D%28x_0%29%5D%3D0 "-E[f(x_0)\hat{f}(x_0)]-E[\epsilon\hat{f}(x_0)]+f(x_0)E[\hat{f}(x_0)]=0")

-   ![E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])(E\[\\hat{f}(x\_0)\]-f(x\_0))\]](https://latex.codecogs.com/png.latex?E%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%28E%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5D "E[(\hat{f}(x_0)-E[\hat{f}(x_0)])(E[\hat{f}(x_0)]-f(x_0))]")

![E\[(\\hat{f}(x\_0)-E\[\\hat{f}(x\_0)\])(E\[\\hat{f}(x\_0)\]-f(x\_0))\]=](https://latex.codecogs.com/png.latex?E%5B%28%5Chat%7Bf%7D%28x_0%29-E%5B%5Chat%7Bf%7D%28x_0%29%5D%29%28E%5B%5Chat%7Bf%7D%28x_0%29%5D-f%28x_0%29%29%5D%3D "E[(\hat{f}(x_0)-E[\hat{f}(x_0)])(E[\hat{f}(x_0)]-f(x_0))]=")

![E\[\\hat{f}(x\_0)E\[\\hat{f}(x\_0)\]-E\[\\hat{f}(x\_0)\]E\[\\hat{f}(x\_0)\]-](https://latex.codecogs.com/png.latex?E%5B%5Chat%7Bf%7D%28x_0%29E%5B%5Chat%7Bf%7D%28x_0%29%5D-E%5B%5Chat%7Bf%7D%28x_0%29%5DE%5B%5Chat%7Bf%7D%28x_0%29%5D- "E[\hat{f}(x_0)E[\hat{f}(x_0)]-E[\hat{f}(x_0)]E[\hat{f}(x_0)]-")

![\\hat{f}(x\_0)f(x\_0)+E\[\\hat{f}(x\_0)\]f(x\_0)\]=](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x_0%29f%28x_0%29%2BE%5B%5Chat%7Bf%7D%28x_0%29%5Df%28x_0%29%5D%3D "\hat{f}(x_0)f(x_0)+E[\hat{f}(x_0)]f(x_0)]=")

![E\[\\hat{f}(x\_0)\]E\[\\hat{f}(x\_0)\]-E\[\\hat{f}(x\_0)\]E\[\\hat{f}(x\_0)\]-E\[\\hat{f}(x\_0)\]f(x\_0)+E\[\\hat{f}(x\_0)\]f(x\_0)=0](https://latex.codecogs.com/png.latex?E%5B%5Chat%7Bf%7D%28x_0%29%5DE%5B%5Chat%7Bf%7D%28x_0%29%5D-E%5B%5Chat%7Bf%7D%28x_0%29%5DE%5B%5Chat%7Bf%7D%28x_0%29%5D-E%5B%5Chat%7Bf%7D%28x_0%29%5Df%28x_0%29%2BE%5B%5Chat%7Bf%7D%28x_0%29%5Df%28x_0%29%3D0 "E[\hat{f}(x_0)]E[\hat{f}(x_0)]-E[\hat{f}(x_0)]E[\hat{f}(x_0)]-E[\hat{f}(x_0)]f(x_0)+E[\hat{f}(x_0)]f(x_0)=0")

최종적인 **expected Test MSE**는 다음과 같이 분해될 수 있다.

![E\[y\_0-\\hat{f}(x\_0))^2\]=Bias^2\[\\hat{f}(x\_0)\]+Var(\\hat{f}(x\_0))+Var(\\epsilon)](https://latex.codecogs.com/png.latex?E%5By_0-%5Chat%7Bf%7D%28x_0%29%29%5E2%5D%3DBias%5E2%5B%5Chat%7Bf%7D%28x_0%29%5D%2BVar%28%5Chat%7Bf%7D%28x_0%29%29%2BVar%28%5Cepsilon%29 "E[y_0-\hat{f}(x_0))^2]=Bias^2[\hat{f}(x_0)]+Var(\hat{f}(x_0))+Var(\epsilon)")

------------------------------------------------------------------------

### Simulation (Regression)

------------------------------------------------------------------------

이제 이에 대한 내용을 직접 시뮬레이션하여 확인해보는 작업을 수행하였다.

**Note :** 위의 Bias와 Variance 는 확률론적 **기댓값**이다. 따라서 이를 추정하기 위해서는 ![N](https://latex.codecogs.com/png.latex?N "N")번의 시뮬레이션을 통해 얻은 ![N](https://latex.codecogs.com/png.latex?N "N")개의 추정치들의 평균으로 구해야하며 따라서 추정치가 **'여러 개'** 여야한다.

이는 결국 같은 차수(Complexity)에 대한 모델 또한 여러 개여야 한다는 말이고 이는 true function을 안다면 그 true function 하의 표본크기 ![n](https://latex.codecogs.com/png.latex?n "n")의 샘플 { (![x\_1](https://latex.codecogs.com/png.latex?x_1 "x_1"),![y\_1](https://latex.codecogs.com/png.latex?y_1 "y_1")),(![x\_2](https://latex.codecogs.com/png.latex?x_2 "x_2"),![y\_2](https://latex.codecogs.com/png.latex?y_2 "y_2")),...,(![x\_n](https://latex.codecogs.com/png.latex?x_n "x_n"),![y\_n](https://latex.codecogs.com/png.latex?y_n "y_n"))} 들을 반복적으로 추출하여 모델을 반복적으로 적합하는 과정이 필요하다는 것을 의미한다.

이를 위해 시뮬레이션을 진행해야하는데 샘플을 얻기 위해 다음과 같이 우리의 true function ![f](https://latex.codecogs.com/png.latex?f "f")를 안다는 가정하에 ![x,y](https://latex.codecogs.com/png.latex?x%2Cy "x,y")와 train 데이터를 generate 하는 process에 대해 정의해야한다.

단계는 이렇다.

**1.** ![y](https://latex.codecogs.com/png.latex?y "y")=![f(x)](https://latex.codecogs.com/png.latex?f%28x%29 "f(x)")+![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon") 의 **true function ![f](https://latex.codecogs.com/png.latex?f "f")를 안다고 가정**하고 주어진 정의역 하에서 Train Set 과 Test Observation을 정의하여 Train Set에서 표본 크기가 ![n](https://latex.codecogs.com/png.latex?n "n")개인 표본을 시뮬레이션 횟수 ![N](https://latex.codecogs.com/png.latex?N "N")번 만큼 반복적으로 추출한다.

**2.** **Polynomial Regression**으로 차수를 1,3,6,12로 늘려가며 4개의 모델을 ![N](https://latex.codecogs.com/png.latex?N "N")번의 매 시뮬레이션 마다 적합한다.

**3.** 차수 별로 Test Observation에 대한 ![N](https://latex.codecogs.com/png.latex?N "N")개의 추정치가 있을 것이다. 차수 별 **Variance** 와 **Bias**, 이에 따른 **MSE**를 계산하여 비교한다.

**Training MSE**, **TEST MSE**, **Bias**와 **Variance** 를 계산하는 함수를 정의한다.

``` r
## MSE function for Training MSE and Expected TEST MSE

mse=function(pred,test){
  return (mean((test-pred)^2))
}

## Bias of estimates

bias=function(pred,test){
  return (mean(pred)-test)
}

## Variance of estimates

variance=function(pred,test){
  return (mean((pred-mean(pred))^2))
}
```

표본 크기 ![n](https://latex.codecogs.com/png.latex?n "n")은 100, 시뮬레이션 횟수 ![N](https://latex.codecogs.com/png.latex?N "N") 은 1000번 수행하기로 하였다.

우리가 다 안다는 가정 하에 true function ![f](https://latex.codecogs.com/png.latex?f "f") 와 irreducible error ![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon")를 다음과 같이 정의하고

![y=0.15~(x-5)^3+\\epsilon,~~(\\epsilon\\sim~N(0,5^2) )~~(0&lt;=x&lt;=10)](https://latex.codecogs.com/png.latex?y%3D0.15~%28x-5%29%5E3%2B%5Cepsilon%2C~~%28%5Cepsilon%5Csim~N%280%2C5%5E2%29%20%29~~%280%3C%3Dx%3C%3D10%29 "y=0.15~(x-5)^3+\epsilon,~~(\epsilon\sim~N(0,5^2) )~~(0<=x<=10)")

 test observation ![(x\_0,y\_0)](https://latex.codecogs.com/png.latex?%28x_0%2Cy_0%29 "(x_0,y_0)")을 ![(8,f(8))](https://latex.codecogs.com/png.latex?%288%2Cf%288%29%29 "(8,f(8))")으로 설정하여

![(x\_0,y\_0)](https://latex.codecogs.com/png.latex?%28x_0%2Cy_0%29 "(x_0,y_0)") 을 제외한 모든 구간에서 train sample을 추출하는 `get_sim` 함수를 정의한다.

``` r
## define true function

f=function(x){
  return (0.15*(x-5)^3)
}

## train data generating function

get_sim=function(f,sample_size=100){
  ## train 
  while (TRUE){
    x=runif(n=sample_size,0,10)
    if (!(8 %in% x)){
      break
    }
    }
  ## epsilon
  epsilon=rnorm(n=sample_size,mean=0,sd=5)
  ## y
  y=f(x)+epsilon
  data=data.frame(x,y)
  return (data)
}
```

위와 같이 정의한 정의역에서 true function ![f](https://latex.codecogs.com/png.latex?f "f") 하의 전체 데이터가 있다고 가정 하면 **대략** 이런 형태가 될 것이다.

``` r
x=seq(0,10,by=0.01)
epsilon=rnorm(length(x),sd=5)
y=f(x)+epsilon
example=data.frame(x,y)
ggplot(example)+
  geom_point(aes(x,y,color='dodgerblue'))+
  geom_line(aes(x,f(x),color='orange'),lwd=1.2)+
  theme_classic()+
  scale_colour_manual(name='',
                      values=c('dodgerblue'='dodgerblue','orange'='orange'),labels=c('y (f(x)+epsilon)','f(x) (true function)'))+
  ggtitle('True function and our data')+
  theme(axis.title=element_text(face='bold'))+
  theme(plot.title=element_text(face='bold'))
```

![](./unnamed-chunk-5-1.png)

이렇게 셋팅을 하고 **sample size**가 **100**인 **1000** (simulation 횟수) 개의 train set ![D](https://latex.codecogs.com/png.latex?D "D")를 generate 했으며 각각의 데이터셋 ![D\_1](https://latex.codecogs.com/png.latex?D_1 "D_1"),![D\_2](https://latex.codecogs.com/png.latex?D_2 "D_2"),...![D\_{1000}](https://latex.codecogs.com/png.latex?D_%7B1000%7D "D_{1000}") 에 대해 1, 3, 6, 12로 차수를 늘려가며 4개의 다항회귀모형을 적합하였고,

Test MSE의 계산을 위한 new observation (![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0"), ![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0")) (![(8,f(8))](https://latex.codecogs.com/png.latex?%288%2Cf%288%29%29 "(8,f(8))")) 를 정의하였으며 예측치들을 `predictions` 이라는 행렬로 저장하였다.

``` r
## number of simulations
n_sim = 1000
## number of models
n_model = 4
degrees=c(1,3,6,12)

## define test set
new_data=data.frame(x=8,y=f(8))


## predictions
predictions=matrix(0,nrow=n_sim,ncol=n_model)
```

이제 각 모델 4개의 ![x\_0 = 8](https://latex.codecogs.com/png.latex?x_0%20%3D%208 "x_0 = 8") 에서의 ![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0") 에 대한 예측치 ![\\hat{y\_0}](https://latex.codecogs.com/png.latex?%5Chat%7By_0%7D "\hat{y_0}")들은 이러한 형태가 될 것이고,

![1th~Degree~of~Polynomial~Regression](https://latex.codecogs.com/png.latex?1th~Degree~of~Polynomial~Regression "1th~Degree~of~Polynomial~Regression") = ![\\hat{f}\_1(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D_1%28x%29 "\hat{f}_1(x)") = ![\\hat{\\beta\_0}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D "\hat{\beta_0}")

![3th~Degree~of~Polynomial~Regression](https://latex.codecogs.com/png.latex?3th~Degree~of~Polynomial~Regression "3th~Degree~of~Polynomial~Regression") = ![\\hat{f}\_3(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D_3%28x%29 "\hat{f}_3(x)") = ![\\hat{\\beta\_0}+\\hat{\\beta\_1}x+\\hat{\\beta\_2}x^2+\\hat{\\beta\_3}x^3](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D%2B%5Chat%7B%5Cbeta_1%7Dx%2B%5Chat%7B%5Cbeta_2%7Dx%5E2%2B%5Chat%7B%5Cbeta_3%7Dx%5E3 "\hat{\beta_0}+\hat{\beta_1}x+\hat{\beta_2}x^2+\hat{\beta_3}x^3")

![6th~Degree~of~Polynomial~Regression](https://latex.codecogs.com/png.latex?6th~Degree~of~Polynomial~Regression "6th~Degree~of~Polynomial~Regression") = ![\\hat{f}\_6(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D_6%28x%29 "\hat{f}_6(x)") = ![\\hat{\\beta\_0}+\\hat{\\beta\_1}x+...+\\hat{\\beta\_{6}}x^{6}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D%2B%5Chat%7B%5Cbeta_1%7Dx%2B...%2B%5Chat%7B%5Cbeta_%7B6%7D%7Dx%5E%7B6%7D "\hat{\beta_0}+\hat{\beta_1}x+...+\hat{\beta_{6}}x^{6}")

![12th~Degree~of~Polynomial~Regression](https://latex.codecogs.com/png.latex?12th~Degree~of~Polynomial~Regression "12th~Degree~of~Polynomial~Regression") = ![\\hat{f}\_{12}(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D_%7B12%7D%28x%29 "\hat{f}_{12}(x)") = ![\\hat{\\beta\_0}+\\hat{\\beta\_1}x+...+\\hat{\\beta\_{12}}x^{12}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D%2B%5Chat%7B%5Cbeta_1%7Dx%2B...%2B%5Chat%7B%5Cbeta_%7B12%7D%7Dx%5E%7B12%7D "\hat{\beta_0}+\hat{\beta_1}x+...+\hat{\beta_{12}}x^{12}")

``` r
set.seed(2013122044)
sim_data=get_sim(f,100)
for (i in degrees){
  poly=lm(y~poly(x,i),data=sim_data)
  pred=predict(poly,newdata=sim_data)
  name=paste0('plot',i)
  assign(name,ggplot(data=sim_data,aes(x=x,y=y))+
           geom_point(col='dodgerblue')+
           geom_line(aes_string(x=sim_data$x,y=pred),color='orange',lwd=1.2)+
           theme_classic()+
    ggtitle(paste0(i,'th Degree of Polynomial Regression Model'))+
      theme(plot.title=element_text(face='bold')))
}

grid.arrange(plot1,plot3,plot6,plot12,
             nrow=2,ncol=2)
```

![](./unnamed-chunk-7-1.png)

차수 별 fitted line은 위의 형태와 같을 것이다.

1000회의 시뮬레이션을 거쳐 확인한 결과는 다음과 같다.

``` r
## Bias - Variance Trade-Off Simulation 
options(scipen=999)
set.seed(2013122044)

## 1000 simulations
for (sim in 1:n_sim){
  
  ## generate dataset with sample size 100
  sim_data=get_sim(f,100)
  
  ## model fitting and predictions
  for (j in 1:length(degrees)){
    fit=lm(y ~ poly(x,degrees[j]),data=sim_data)
    predictions[sim,j]=predict(fit,newdata=new_data)
  }
}

predictions=data.frame(predictions)
colnames(predictions)=c('1th','3th','6th','12th')


ggplot(data =melt(predictions),aes(x=variable, y=value))+
  geom_jitter(aes(color=variable),width=0.3)+
  geom_boxplot()+
  theme_classic()+
  geom_hline(yintercept = f(8),color='red',lwd=1)+
  xlab('Degrees')+
  ylab('Values')+
  theme(legend.position="none")+
  ggtitle('Boxplot of Predictions')+
  theme(axis.title=element_text(face='bold'))+
  theme(plot.title=element_text(face='bold'))
```

    ## No id variables; using all as measure variables

![](./unnamed-chunk-8-1.png)

빨간선이 true function 하의 ![f(x\_0)= f(8)](https://latex.codecogs.com/png.latex?f%28x_0%29%3D%20f%288%29 "f(x_0)= f(8)") 값이고 점들은 각 다항회귀의 차수에 대한 모델의 예측값을 나타낸다.

각각의 차수에 대해 Test observation 에 대한 **Test~MSE**와 **Bias^2**, **Variance**를 계산한 결과는 다음과 같다.

``` r
## Test MSE
test_mse=apply(predictions,2,mse,test=f(8))

## Bias^2
bias_squared=(apply(predictions,2,bias,test=f(8)))^2

## Variance
var=apply(predictions,2,variance,test=f(8))

results=data.frame('Test MSE'=test_mse,'Bias Squared'=bias_squared,'Variance'=var)

kable(results,format='markdown') 
```

|      |   Test.MSE|  Bias.Squared|   Variance|
|:-----|----------:|-------------:|----------:|
| 1th  |  7.9000519|     7.1575396|  0.7425123|
| 3th  |  0.7576468|     0.0017169|  0.7559299|
| 6th  |  1.3258296|     0.0003601|  1.3254694|
| 12th |  2.9330009|     0.0000029|  2.9329980|

3차 다항회귀를 적합했을때, 우리가 알고있다고 가정한 true function ![f](https://latex.codecogs.com/png.latex?f "f") 가 3차함수였으므로 Test MSE가 가장 작다고 유추할 수 있다.

**Note :** 정의한 함수로 계산되는 Test MSE를 보면 정확히 Bias의 제곱과 Variance의 합임을 알 수 있다.

정의대로면 추가적으로 ![Var(\\epsilon)](https://latex.codecogs.com/png.latex?Var%28%5Cepsilon%29 "Var(\epsilon)") 이 고려되어야 하지만 이는 우리가 시뮬레이션을 위해 new observation ![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0") 값을 안다고 가정했기 때문에 아직 관측되지 않은 ![y\_0](https://latex.codecogs.com/png.latex?y_0 "y_0") 값 자체가 가지는 irreducible error (![\\epsilon](https://latex.codecogs.com/png.latex?%5Cepsilon "\epsilon"))을 고려하지 않았기 때문이다.

------------------------------------------------------------------------

### Simulation (Classification)

------------------------------------------------------------------------

위의 회귀 문제가 아닌 분류 문제일때는 어떨까?

이 경우는 Train Observations { ![(x\_1,y\_1)](https://latex.codecogs.com/png.latex?%28x_1%2Cy_1%29 "(x_1,y_1)") ,,, ![(x\_n,y\_n)](https://latex.codecogs.com/png.latex?%28x_n%2Cy_n%29 "(x_n,y_n)") } 로 똑같이 ![f](https://latex.codecogs.com/png.latex?f "f") 에 대해 추정하는 것이다. 다만 종속변수 ![y](https://latex.codecogs.com/png.latex?y "y") 가 **qualitative** (질적변수) 인 것이다.

이 경우 모델의 정확도를 평가하는 지표로 \*\* Training Error Rate \*\* 를 자주 사용하는데 이는 우리의 추정 모델 ![(\\hat{f})](https://latex.codecogs.com/png.latex?%28%5Chat%7Bf%7D%29 "(\hat{f})") 가 train observation들에 대해 맞추지 못한 비율로 모델의 정확도를 평가한다.

![\\frac{1}n\\Sigma~I(y\_i\\ne\\hat{y\_i})](https://latex.codecogs.com/png.latex?%5Cfrac%7B1%7Dn%5CSigma~I%28y_i%5Cne%5Chat%7By_i%7D%29 "\frac{1}n\Sigma~I(y_i\ne\hat{y_i})")

``` r
## error rate

error_rate=function(pred,test){
  mistakes=c()
  for (i in 1:length(test)){
    if (pred[i]!=test[i]){
      mistakes[i]=1
    }
    else{
      mistakes[i]=0
    }
  }
  return (mean(mistakes))
}
```

위의 수식에서 ![I(y\_i\\ne\\hat{y\_i})](https://latex.codecogs.com/png.latex?I%28y_i%5Cne%5Chat%7By_i%7D%29 "I(y_i\ne\hat{y_i})") 은 *indicator variable* 로 예측치와 관측치가 같다면 0, 다르다면 1로써 error rate를 계산할 것이다.

위의 회귀 문제처럼 **Test Error Rate** 를 최소화하기 위한 과정에서 분류 문제 또한 **Bias & Variance Trade Off** 가 적용된다.

책에서는 Bayes Classifier 와 KNN에 대한 아주 간단한 설명과 함께 두 가지 방법론에 대해 이를 설명하고 있기에 직접 확인해보았다.

------------------------------------------------------------------------

#### Bayes Classifier

------------------------------------------------------------------------

매우 단순한 분류기에 해당하는 베이즈 분류기는 주어진 각각의 독립변수 값에 대해 **가장 그럴듯한** (maximum likelihood) class에 종속 변수 값을 할당하는 **조건부 확률 모델**이다.

즉, 만일 class가 1과 2로 binary classification 문제라고 가정하면 베이즈 분류기는 조건부 확률은 주어진 독립변수값 ![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0") 하에서 ![Pr(Y=1|X=x\_0)&gt;0.5](https://latex.codecogs.com/png.latex?Pr%28Y%3D1%7CX%3Dx_0%29%3E0.5 "Pr(Y=1|X=x_0)>0.5") 라면 class 1으로 분류하는 것이고, 아니라면 2로 분류하는 것이다.

베이즈 분류기는 이론적인 **실제 조건부 확률 분포** ![Pr(Y=i|X=x\_0)](https://latex.codecogs.com/png.latex?Pr%28Y%3Di%7CX%3Dx_0%29 "Pr(Y=i|X=x_0)") 를 알기 때문에 어떠한 Test set 이 주어져도 Test error rate를 최소화하는 **규칙**이고 이를 따라서 **Bayes error rate** 라고 한다. 즉, 책에서 말하는 \*\*irreducible error\*에 해당한다고 할 수 있다.

즉, 베이즈 분류기는 위의 **Regression** 의 예시에서 제시한 일종의 true function 을 안다고 할 수 있는 확률 모형이기 때문에 Validation 의 과정도 필요없고 영원히 도달할 수 없는 궁극의 목표같은 것이다.

베이즈 분류기를 쓴다면 참 좋겠지만 실제로 우리는 우리의 observation 들이 generate되는 process (true function) 에 대해 알지 못하므로 조건부 확률 ![Pr(Y=i|X=x\_0)](https://latex.codecogs.com/png.latex?Pr%28Y%3Di%7CX%3Dx_0%29 "Pr(Y=i|X=x_0)") 를 계산하는 것이 불가능하다. 따라서 이러한 조건부 확률을 추정하여 클래스에 속할 **추정된 확률**이 가장 큰 클래스로 분류 해야하는데 조건부 확률을 추정하는 방법이 여러가지가 있고 이 중 하나가 KNN이다.

------------------------------------------------------------------------

#### KNN (K-Nearest Neighbors)

------------------------------------------------------------------------

KNN은 Test observation ![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0") 이 주어졌을 때 ![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0") 근처의 가장 가까운 값 ![K](https://latex.codecogs.com/png.latex?K "K") 개 관측치를 Train data에서 찾는다.

그 관측치들 중 class가 j인 부분들이 있을 것이고, 그 다음 class j에 속할 조건부 확률들을 K개의 이웃들로 계산하는 것이다. 아래와 같이 계산한다.

![Pr(Y=j|X=x\_0)=\\frac{1}{K}\\Sigma~I(y\_i=j)](https://latex.codecogs.com/png.latex?Pr%28Y%3Dj%7CX%3Dx_0%29%3D%5Cfrac%7B1%7D%7BK%7D%5CSigma~I%28y_i%3Dj%29 "Pr(Y=j|X=x_0)=\frac{1}{K}\Sigma~I(y_i=j)")

최종적으로 이렇게 추정된 조건부 확률들로 베이즈 정리를 이용하여 test observation ![x\_0](https://latex.codecogs.com/png.latex?x_0 "x_0") 를 가장 속할 확률이 높은 class에 분류하는 것이다.

예를 들면, ![K](https://latex.codecogs.com/png.latex?K "K") 가 1이면 무조건 제일 가까운 관측치와 같은 클래스로 분류될 것이고 Train observation이 100개인데 ![K](https://latex.codecogs.com/png.latex?K "K") 가 100이면 전체 Train data에서 제일 많은 클래스로 분류해버릴 것이다.

이웃 한명 말만 듣고 결정하는 것과 100명 말을 듣고 결정하는 것, KNN의 경우 또한 모델의 복잡도에 따른 **Bias-Variance Trade off** 가 존재할 것이다.

`ElemStatLearn` 패키지에 있는 실습 자료 `mixture.example` 을 이용해 K값을 달리해보며 분류 모델을 적합한 결과는 다음과 같다.

**Note** : 확률 분포를 가정하여 generate한 데이터지만 실습을 위해 확률분포를 모른다고 가정하고 KNN 모델을 적합하였다.

먼저 데이터셋에는 조건부 확률값들이 주어져 있다. 따라서 이를 토대로 **Bayes Decision Boundary**를 나타낸 결과는 다음과 같다.

``` r
## data

X1_train=mixture.example$x[,1]
X2_train=mixture.example$x[,2]

Y=mixture.example$y
Y=factor(Y)

knn_train=data.frame(X1=X1_train,X2=X2_train,Y=Y)
knn_test=data.frame(X1=mixture.example$xnew[,1],X2=mixture.example$xnew[,2])

px1=mixture.example$px1
px2=mixture.example$px2
prob=mixture.example$prob
prob=matrix(prob, length(px1), length(px2))
bdb=data.frame(expand.grid(x=px1, y=px2),c(prob))
colnames(bdb)=c('x','y','prob')

ggplot(knn_train)+
    geom_point(aes(x=X1,y=X2,color=Y),size=2)+
    stat_contour(data=bdb,aes_string(x=bdb$x,y=bdb$y,z=bdb$prob),color='black',lwd=1.2,lty='dashed',breaks=0.5)+
    scale_colour_manual(name='', values=c('0'='cornflowerblue','1'='coral'),labels=c('0','1'))+
    theme_classic()+
    ggtitle(paste0('Bayes Decision Boundary'))+
  theme(plot.title=element_text(face='bold'))
```

![](./unnamed-chunk-13-1.png)

또한, 조건부 확률 분포가 주어져 있으므로 주어진 ![X](https://latex.codecogs.com/png.latex?X "X") 하에 확률이 가장 큰 ![Y](https://latex.codecogs.com/png.latex?Y "Y") 를 선택했을 때 발생하는 오차값인 **Bayes Error Rate** 는 대략 0.1232의 값을 가진다.

``` r
## Bayes error rate
set.seed(2013122044)

y_bayes=rep(0,6831)
for (j in 1:6831){
  if (bdb$prob[j] > 0.5){
    y_bayes[j]=1
  }
  else if (bdb$prob[j] < 0.5){
    y_bayes[j]=0
  }
  else {
    y_bayes[j]=sample(c(1,0),1,prob=c(bdb$prob[j],1-bdb$prob[j]))
  }
}

bayes_error=c()
for (i in 1:100){
  y_real=rep(0,6831)
  for (j in 1:6831){
    y_real[j]=sample(c(1,0),1,prob=c(bdb$prob[j],1-bdb$prob[j]))
  }
  bayes_error=append(bayes_error,error_rate(y_bayes,y_real))
}

bayes_error=mean(bayes_error)
```

위의 자료로 K값을 1,10,50,100까지 늘려가며 KNN 분류 모델을 적합해 **KNN Decision Boundary** 을 나타낸 결과는 다음과 같다.

``` r
## fit
ks=c(1,10,50,100)
for (i in ks){
  model=knn(knn_train[,-3],knn_test,knn_train$Y,k=i,prob=TRUE)
  prob=attr(model, "prob")
  prob=ifelse(model=="1", prob, 1-prob)
  px1=mixture.example$px1
  px2=mixture.example$px2
  prob=matrix(prob, length(px1), length(px2))
  kdb=data.frame(expand.grid(x=px1, y=px2),c(prob))
  colnames(kdb)=c('x','y','prob')

  name=paste0('k',i)
  assign(name,ggplot(knn_train)+
    geom_point(aes(x=X1,y=X2,color=Y),size=2)+
    stat_contour(data=kdb,aes_string(x=kdb$x,y=kdb$y,z=kdb$prob),color='black',lwd=1.2,lty='dashed',breaks=0.5)+
    scale_colour_manual(name='', values=c('0'='cornflowerblue','1'='coral'),labels=c('0','1'))+
    theme_classic()+
    ggtitle(paste0('K=',i))+
      theme(axis.title=element_text(face='bold'))+
      theme(plot.title=element_text(face='bold'))
      
  )
}

grid.arrange(k1,k10,k50,k100,nrow=2,ncol=2)
```

![](./unnamed-chunk-15-1.png)

이제 데이터셋에 주어진 조건부 확률분포 `prob` ![(Pr(Y=1|x\_0=\[x\_1~x\_2\]^T))](https://latex.codecogs.com/png.latex?%28Pr%28Y%3D1%7Cx_0%3D%5Bx_1~x_2%5D%5ET%29%29 "(Pr(Y=1|x_0=[x_1~x_2]^T))") 을 이용해 500개의 test observation을 generate 해서 K를 1부터 200까지 설정할 때마다 **Training Error Rate** 와 **Test Error Rate** 가 어떻게 변하는지 확인해본 결과는 다음과 같다.

``` r
## test set generation
set.seed(2013122044)
X1=sample(px1,500,replace=TRUE)
X2=sample(px2,500,replace=TRUE)

idx=c()
for (i in 1:500){
  idx=append(idx,which(bdb$x==X1[i] & bdb$y==X2[i]))
}

prob=bdb$prob[idx]

## test y

y=rep(0,500)
for (i in 1:500){
  y[i]=sample(c(1,0),1,prob=c(prob[i],1-prob[i]))
}

## test df

knn_test=data.frame(X1=X1,X2=X2,Y=y)

##training error rate and test error rate

train_error=c()
test_error=c()


for (i in 1:200){
  train=knn(knn_train[,-3],knn_train[,-3],knn_train$Y,k=i,prob=TRUE)
  test=knn(knn_train[,-3],knn_test[,-3],knn_train$Y,k=i,prob=TRUE)
  train_error=append(train_error,error_rate(train,knn_train$Y))
  test_error=append(test_error,error_rate(test,knn_test$Y))
}
## results df
K=1:200
error=c(rep('Train Error',200),rep('Test Error',200))
results=data.frame(c(train_error,test_error),error,K)
colnames(results)=c('Value','Error','K')

ggplot(results)+
  geom_line(aes(x=K,y=Value,color=Error),lwd=1.2)+
  theme_classic()+
  scale_colour_manual(name='', values=c('Test Error'='cornflowerblue','Train Error'='coral'),labels=c('Test Errors','Training Errors'))+
  geom_hline(yintercept = bayes_error,lwd=1,lty='dashed')+
  ylab('Error Rate')+
  geom_vline(xintercept = which(test_error==min(test_error)),color='red',lwd=1,lty='dashed')+
  geom_text(aes(x=15, label="K=5", y=0.05), colour="red",size=5)+
  ylab('Error Rate')+
  ggtitle('KNN Training Error Rate vs. Test Error by K')+
  theme(axis.title=element_text(face='bold'))+
  theme(plot.title=element_text(face='bold'))
```

![](./unnamed-chunk-16-1.png)

위 그래프에서 검은색 점선이 바로 **Bayes Decision Boundary** 이고, 빨간색 점선이 Test Error가 최솟값을 갖는 **K=5** 를 나타낸다.

이렇게 회귀와 분류 문제 모두 모델의 적절한 복잡도를 선택하는 것이 좋은 통계 학습 모델의 필수 조건이라는 것을 확인하였다.

첫단원이라 추정량이나 모델에 대한 자세한 설명이 없어서 이해가 부족한 부분이 있으나 직접 이론을 공부하고 구현해보며 더 깊은 이해를 할 수 있었다.
