# linear regression 이해하고 Gradient descent로 직접 최적화하기

> (가독성과 재생산성을 모두 살리기 위해 맨 아래부분에 직접사용한 함수들을 모아놓았습니다. 코드를 실행하려면 맨아래 cell의 함수를 먼저 실행하고 위에서 부터 순서대로 실행하면 됩니다.)

**아직 읽은 부분이 많지도 않아 충분히 바뀔 수 있습니다만, 한주동안 책을 접해보니, 다음과 같습니다.**

1. 가볍게 써져있다. 모델에 대한 근원적인 이야기나 수학적인 원리등은 거의(isl보다도 더) 안건들고 있습니다. 모델에 대한 간단한 특성설명, 파이썬으로 구현해보기가 메인입니다.


2. 그럼에도 불구하고 좋다. ISL로 모델에 대한 아주 얉은 지식이나마 쌓은 후에 접하기 좋은것 같다. 간단히 설명되는 특성에서도 나름 스스로 얻고 구글링해서 더 알아볼만한 점들이 많이 나온다. 무엇보다, numpy도 없이 모델을 구현하는데 나름 재미도 있다. (for문 돌리면서 멍때리고 쳐다보기..)


# 1. 미분으로 Simple Linear Regression 적합하기

ISL때와 마찬가지로, linear regression부터 나가도록 하겠습니다. 지난 ISL때 선형회귀의 이론에 집중하였다면 이번에는 좀더 선형회귀의 특성과 **gradient descent를 통한 직접적인 구현**에 집중하도록 하겠습니다.

선형회귀는 설명변수와 반응변수간에 선형적인 관계가 있을것이라는 다소 맹목적인 가정하에서 만들어진 모델이다. 고로 설명변수$$x$$와 반응변수 $$y$$가 있을때 목표는 $$ y_i=\beta x_i + \alpha + \epsilon_i$$ 식의 계수를 찾는것으로 귀결된다.


```python
#즉 이런식
def predict(alpha, beta, x_i):
    return beta * x_i + alpha
```

어떻게 최적의 계수 $\alpha, \beta$를 찾을까? 가장 대표적인 방법이 sum of squared error를 줄이는 least square이다. 


```python
def error(alpha, beta, x_i, y_i):
    """the error from predicting beta * x_i + alpha
    when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
```

## 1-2. least square와 maximum likelihood.

least square에 대한 해석이 물론 여러가지 있지만 그 방법중 하나로 **정규가정일경우 maximum likelihood estimator를 찾는것과 같은 결과**가 된다는 것이다.

likelihood는 간단히 말해, 해당 데이터를 가질만한 가장 likely한 모수$$\theta$$이다. (데이터 $v_1,..,v_n$이 있을때, 그들이 나왔을 가장 가능성이 높은 모수 $\theta$)

![likelihood](https://user-images.githubusercontent.com/31824102/44249323-dac5ae00-a1de-11e8-96cb-89d16d63439c.PNG)

근데 만약 우리가 (대표적으로 simple linear에서 그러하듯이, )error에 대해 mean0의 정규분포를 가정하면, $$E(y)=\alpha+\beta*x$$이기 때문에

$$y$$ ~ $$N(\hat y,\sigma)$$이므로 해당 데이터에 대한 pdf는 밑에 식이 된다. (나머지는 fixed x에 대해 모두 fixed이기 때문에 error의 분포를 그대로 받게 된다.)

![mle_mse](https://user-images.githubusercontent.com/31824102/44249329-de593500-a1de-11e8-91e6-794bdb47c397.PNG)

또한, 만약 우리가 (대부분의 ML 기법이 기본가정으로 깔고가듯이) **각 데이터가 서로 독립이라고 가정**한다면,  전체 데이터에 대한 joint_pdf는 데이터 n개에 대해 곱하는 것과 같다. 수식으로 나타내면 다음과 같다.


$$
\begin{align} J(\alpha,\beta)&=\prod_i^n\frac{1}{\sqrt{2\pi\sigma}}exp(\frac{-(y_i-\alpha-\beta x_i)^2}{2\sigma^2})\\ & =(\frac{1}{\sqrt{2\pi\sigma}})^nexp(\frac{-\sum_i^n(y_i-\alpha-\beta x_i)^2}{2\sigma^2})\end{align}
$$

이때, $\pi$나 $\sigma$같은 (모델의 가정하에서) fixed constant를 빼면, 결국 $$\sum(y-\alpha-\beta x)^2$$, 즉 **least square만이 남는다**. 다시말해, error에 대한 정규분포를 가정하였을 경위 least square는 mle를 찾는 것과 완벽하게 동치이다.

다시 본론으로 돌아와서, leas square를 봐보자. 

위 식에서 loss fuctiondl, 즉 sum of squares를 최소화하는 계수는 다음과 같다. 

$b_{0}=\bar y -b_{1}\bar x$

$b_{1}=\frac{\sum x_{i}(y_{i}-\bar y)}{\sum x_{i}(x_{i}-\bar x)}$

> **$b_{0}$의 경우**
>
> $$\frac {\partial RSS} {\partial b_0}=\frac {\partial \sum_{i=1}^{n}(y_{i}-b_{0}-b_{1}x_{i})^2} {\partial b_0}=0$$이 되는 값을 찾으면 된다.
>
> ${- 2\sum_{i=1}^{n}(y_{i}-b_{0}-b_{1}x_{i})}=0 $
>
> ${\sum_{i=1}^{n}(y_{i}-b_{0}-b_{1}x_{i})}=0 $
>
> ${\sum(y_{i})-nb_{0}-\sum b_{1}x_{i}}=0 $
>
> ${\sum(y_{i})-\sum b_{1}x_{i}}=nb_{0} $
>
> $b_{0}=\frac{\sum(y_{i})}{n} -\frac{\sum b_{1}x_{i}}{n}$
>
> $\therefore b_{0}=\bar y -b_{1}\bar x$
>
> **$b_{1}$의 경우**
>
> $$\frac {\partial RSS} {\partial b_1}=\frac {\partial \sum_{i=1}^{n}(y_{i}-b_{0}-b_{1}x_{i})^2} {\partial b_1}=0$$이 되는 값을 찾으면 된다.
>
> ${- 2\sum_{i=1}^{n}x_{i}(y_{i}-b_{0}-b_{1}x_{i})}=0 $
>
> ${\sum_{i=1}^{n}x_{i}(y_{i}-b_{0}-b_{1}x_{i})}=0 $
>
> ${\sum(x_{i}y_{i})-b_{0}\sum x_{i}-\sum b_{1}x_{i}^2}=0 $
>
> $\sum x_{i}(y_{i}-\bar y)-b_{1}\sum x_{i}(x_{i}-\bar x)=0, \because b_{0}=\bar y -b1\bar x$
>
> $$\therefore b_{1}=\frac{\sum x_{i}(y_{i}-\bar y)}{\sum x_{i}(x_{i}-\bar x)}$$, 이는 $$\frac{\sum (x_{i}-\bar x)(y_{i}-\bar y)}{\sum (x_{i}-\bar x)(x_{i}-\bar x)} $$로도 나타낼 수 있어(전개하면 똑같다) $$b_{1}=\frac{Sxy}{Sxx}$$라고 쓰기도 한다 (sum of x&y, sum of x&x)

식을 잘보면, 결국 $\beta$(혹은 $b_1$)는 $\frac{cov(x,y)}{var(x)}$임을 알 수 있다.(분모분자에 각각 n혹은 n-1이 약분되었다고 보면 된다.) 

따라서 1차 단순선형회귀에서 $\beta$는 **x변수의 변동성 대비 y와 관계있는 정도**라고 정성적으로 이해하여 쓰이기도 한다.(사회과학에서는 특히나 이런식으로 자주 쓰인다.) 

파이썬에서 미분을 전부 해서 스스로 최적해를 찾도록 할 수 있다면 좋겠지만, 파이썬에서 미분은 가능하지 않다. 그래서 이 책에서는 least_square_fit이라는 함수에 단순히 cov/var로써 표현을 하였다.(cov/var는 $corr*std(y)/std(x)$와 똑같다. corr의 정의가 $\frac{cov}{std(x) * std(y)}$이므로.)


```python
def least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
```

이를 통해 사이트내 친구 수로써 사이트활동 시간을 예측하는 선형회귀모델을 만들어보자.


```python
num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
print("alpha", alpha)
print("beta", beta)

```

    alpha 22.9167813834
    beta 0.908340529495

결과는 수식으로 다 정리해놓고 cov/var로 파라미터 $\beta$들을 그냥 적합을 해버리니 조금 심심하다.  좀더 스스로 짠 모델답도록 알아서 최적의 해를 찾아가게 할 방법은 없을까?

# 2. Gradient descent

이번엔 그럼 이미 계산된 해로써 모델을 만드는게 아니라, gradient descent를 사용하여 직접 최적화를 시켜주자. 그전에, gradient descent에 대해서 알아보자.

###Gradient descent란? 

Gradient descent는, 대표적인 parameter 최적화방법으로, 매 iteration(혹은 시도)마다 각 parameter들의 gradient를 구해서 loss func를 최소화하는 방향으로 업데이트 해주는 최적화방식이다.(말이 조금 지저분한데, 아래 다시 설명한다.)

우리가 만든 모델은 어떤 목적함수(cost function)이 있을테고, 그를 구성하고 있는 파라미터$\beta$들이 있을 것이다. (위의 linear regression예시에서는 $\sum_{i=1}^{n}(y_{i}-b_{0}-b_{1}x_{i})^2$, 즉 sse가 우리의 목적함수 였고 그를 구성하는 파라미터는 각각 기울기$b_1$과 intercept $b_0$이 있었다.). cost function은 사실상 $b_0,b_1...$등의 **변수**로 이루어진 **함수**이므로, 우리가 이차함수 $y=x^2$을 그리듯이, 아래의 그림처럼 그래프로써 표현할 수 있다(절대로 아래 그림처럼 항상 원만한 형태로 나오지는 않을것이다). 이때 cost function을 최소화 시키는 parameter들을 찾는것이 우리의 목적이다. 변수가 하나뿐인 simple linear regresssion의 경우 몇줄의 수식으로 손쉽게 미분을하여 cost function을 최소화시키는 점을 찾을 수 있었기에, 이 경우는 굳이 다른 최적화 방법을 찾을 필요가 없다. 그러나, 변수가 많아지거나, cost function이 복잡해지는 등 문제가 조금만 복잡해지면, 최적해를 찾기가 굉장히 어렵거나, 혹은 최적해가 존재하는지도 명확하지 않은 경우가 많다. 그 경우 가능한 최적해에 가깝(근사)하다고 할만한 해를 찾아야 하는데, 이때 여러가지 최적화방법이 사용된다.

![gradient](https://user-images.githubusercontent.com/31824102/44206560-ca172880-a149-11e8-8221-0298002571b0.PNG)

gradient descent는 cost function의 최적해를 찾기위해, 각각의 계수에 대하여 미분(혹은 다변량인 경우 편미분)을 하여 gradient를 구하고, 해당 cost function을 최소화 시키는 방향으로 조금씩 update를 해나가는 최적화 방법이다. 

다시 위 그림의 cost function에서, 빨간색 화살표가 gradient방향을 가르키고 해당 방향으로 조금씩 update를 하는 것이다.

### 편미분이란?

편미분이란, 하나의 계수(혹은 변수)에 대해서만 행해지는 미분이다. 예를들어 변수가 $\beta_0,..,\beta_p$와 같이 여러개일때 $\beta_j$의 편미분이란 $\beta_j$를 제외한 모든 변수들이 다 고정된 상수(그 값을 모르더라도, 고정된 채라고 가정하고)라고 치부한 상태에서 $\beta_j$에 대해서만 행해지는 미분이다. 즉, **다른 모든 변수들이 고정되어 있을때,** **해당 변수 $\beta_j$가 아주 조금 변화하였을때, 해당 cost function은 얼마나 영향을 받아 변화할지**를 나타낸다고 할 수 있다.

그럼 편미분은 어떻게 구현하느냐? 단변수에 대한 미분은 아래와 같은 그림처럼, h->0에 수렴하는 아주 작은 값을 움직일때 값이 얼마나 변하는지를 의미한다 할 수 있다.
![approximate](https://user-images.githubusercontent.com/31824102/44206561-ca172880-a149-11e8-8153-3ee2632004de.PNG)
그러나 파이썬에는 limit(0은 아니지만 0에 무한히 가까운 수)를 구현할 수 없기에, 여기서는 아주 작은 값을 input으로 근사를한다


```python
def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):#'h'만큼 움직였을때의 변화율로써 근사를 한다. 여기서 h는 1e-9와같이 매우 작은 수가 들어간다.
    return (f(x + h) - f(x)) / h
```

또, 다변량인 경우 gradient는 편미분을 해서 구하는 함수를 짠다. 1~j개의 변수가 있을때, 각각 다른 변수들은 고정된채 하나의 변수만 아주 조금 옴겼을때의 변화율을 모두 구한다. 


```python
#다변량일때의 gradient는 편미분을해서 구한다.
def partial_difference_quotient(f, v, i, h):#여기서도 역시 'h'만큼 움직였을때의 변화율로써 근사를 한다. 여기서 h는 1e-9와같이 매우 작은 수가 들어간다.
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
        for j, v_j in enumerate(v)]
    # [원래a,원래b,,,,i+h,...]이런식의 리스트 반환, 즉 i만 h만큼 아주 조금 옴겻을때!
    return (f(w) - f(v)) / h #i만 아주 약간의 차이가 있는 두 리스트를 각각의 f(x)에 넣엇을때의 차이(를 h로 나누기)
```

최종적으로 estimate_gradient에는, 편미분을 통해 각각의 변수에 대해 구해진 모든 변화율들을 포함한 list를 반환하는 함수를 만든다.
```python
def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
        for i, _ in enumerate(v)]
#여러 변수의 각각의 편미분이 나오는 리스트 반환
```

이제 간단한 loss funcstion을 정의해서 위의 함수를 test해보자.

예를 들어, loss function을 $$\sum^n_{i} i*{x_i}$$ 이라고 정의해보면 다음과 같다. ($$0*x_0+1*x_1+..10*x_{10}$$과 같은 형태가 될것이다.)

> 사실 loss function에는 실제 데이터를 통해 구해진 $x_i$들이 있을테고, 그를 각 데이터들에 대해 sum(혹은 mean)을 하여 cost function을 구하게 된다. 각 데이터들에 대해서 구하는 작업은 잠시 차치하고, 여기서는 쉬운 이해를 위해 $x_i$를 직접 설정하여 넣어준다.


```python
def my_sum(inp_ilst):
    res_list=[]
    for x in range(len(inp_ilst)):
        res_list.append(inp_ilst[x]*x)
    return res_list,sum(res_list)
```


```python
my_sum([x for x in range(10)])
```


    ([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], 285)
우선 partial_difference_quotient로 5번째 변수에 대해서 편미분값을 구해보자.

```python
#5번째에 대한 편미분
partial_difference_quotient(lambda x:my_sum(x)[1],[777*x for x in range(10)],5,1e-10)#1*i_1+2_i2....식으로 가니, i_5의 편미분은 5!
```


    4.94765117764473
 $0*x_0+1*x_1+..10*x_{10}$와 같은 형태이니 $x_5$의 편미분값이 5가 나오는 것을 확인할 수 있다.(실제 limit값을 사용하지 않았기에, 약간의 계산오차가 존재한다.)

이번에는 etimate_gradient함수를 이용해서, 모든 변수에 대해서 gradient를 구해보자.  $$0*x_0+1*x_1+..10*x_{10}$$와 같은 형태이니 $$x_n$$에 [0~10]의 리스트를 넣거나 [0~20]의 리스트를 넣거나 gradient값은 [0,1,..,10]이 나올 것이다.

```python
estimate_gradient(lambda x:my_sum(x)[1],[2*x for x in range(10)],h=1e-4)
```


    [0.0,
     0.9999999997489795,
     1.999999999497959,
     2.9999999992469384,
     4.000000000132786,
     4.999999999881766,
     5.999999999630745,
     7.000000000516593,
     8.000000000265572,
     9.000000000014552]

이제 gradient는 구했고, 이를 통해 진짜 gradient descent를 이용한 최적화를 해보자. 우선 가장 쉽게, loss func이 $$\sum x^2$$라고 해보자. (해당 loss function식을 최소화하는 $$x$$들은 당연히 $$x=(0,0,.,0)$$일것임을 직관적으로 알고 있다.) 우선 각각에 대한 편미분은 $$2*x$$일 것이므로 이를 정의한다. 


```python
def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
        for v_i, direction_i in zip(v, direction)]
def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]
#다변량 square일때 각각의 편미분들
```


```python
def distance(v,w):
    return np.sum(np.power(np.array(v)-np.array(w),2))
```


```python
# pick a random starting point
v = [random.randint(-10,10) for i in range(3)]
v
```


    [7, 0, 4]
그리곤 구해진 gradient의 방향으로 (**-1** x step_size)만큼 업데이트를 해준다. 만약 더이상 유의미한 업데이트가 되지 않는다 생각하면, break구문을 이용해 멈춘다. (별다른 업데이트가 되지 않았다는 것을 여기서는 update이전의 parameter와 update된 parameter들간의 L2norm으로 정의하였다.)

```python
tolerance = 0.00000000000000000001
while True:
    gradient = sum_of_squares_gradient(v) # compute the gradient at v
    next_v = step(v, gradient, -0.01) # take a negative gradient step
    if distance(next_v, v) < tolerance: # stop if we're converging
        break
    v = next_v # continue if we're not
v
```


    [4.290199788520775e-09, 0.0, 2.4515427362975787e-09]
역시나 0에 매우 근사한 수치로 수렴함을 알 수 있다.

## 왜 "-1 X step_size"???

위에서, gradient를 구하고, (그 gradient의 방향) x (**-1** x step_size)만큼을 업데이트 해준다고 하였다. step_size는 우리가 흔히 아는 learning rate, 즉 그 방향으로 어느정도를 갈지 설정해준것인데, 왜 **-1**을 곱해줄까?

### 임의의 cost function에 대하여, 해당 function을 minimize하는 방향언 '언제나' gradient의 '반대 방향'이다.

사실 gradient descent를 보면서 언제나 $-1*\eta$만큼 업데이트한다는걸 수식으로 보고 별 생각 없이 넘겼었는데, 예전에 누군가 'gradient가 음수이면 -1곱하는게 더 커지는게 아니에요?'하고 물어보니 흠칫한적이 있었다. 왜 gradient의 반대 방향으로 가는게 **'언제나'** minimize의 방향인걸까? 사실 위의 질문은, 'gradient의 뱡향'과 'gradient방향으로 갔을때 cost function의 변화'에 대해 혼동된 질문이다. 

우리의 목적은 **cost function이 얼마나 움직이는 가**이다. 특정 gradient에 대해서, 우리가 어떻게 움직이던 업데이트는 **'움직인 방향'  x 'gradient '**가 될것이다(현재 주어진 정보인 gradient가 유지된다는 가정에서. 즉 taylor expension). 즉,   gradient가 음수일 경우 gradient방향으로 가는 $(x+\Delta x)​$는 '**그negative방향**'으로 '**negative만큼**'간다는 의미가 되버린다. 즉, **gradient방향**으로 갈 경우 **언제나 cost function은 최대화 된다**. 다음의 예시에서 조금 더 설명한다.

![x_square](https://user-images.githubusercontent.com/31824102/44249345-f466f580-a1de-11e8-9c87-bb582806a71d.PNG)

위와 같이 $y=x^2$이라는 함수가 있다고 해보자. gradient는 언제나 $2*x$일 것이다. 만약 데이터 x가 -2였어서 gradient가 -4였다면, 해당 점(즉, -2)에서 gradient만큼 -4를 움직이는 것은 언제나 해당 함수를 증가시킨다.

>  x-4이때 $\Delta x$는 -4, 근데 이때 또 기울기는 -4니가 사실상 업데이트량은 -4*-4즉 **언제나 제곱텀**. 즉 **절대 음수될수 없다**. 즉 언제나 gradient만큼 빼주는게 minimize해주는 방향이다.
>
> 다시 정리하자.
>
> x+1,x 둘의 delta는 1, 이때의 변화는 1*gradient_x이지만,
>
> x+gradient_x까지 간다면 이때의 **delta**는 __gradient_x__*gradient_x. 즉, 제곱항. 항상 양수일수밖에 없다.(x=$$\Delta x$$의 delta는 $$\Delta x$$X$$\Delta x$$) 고로 x+gradient_x는 언제나 maximize하는 방향, x-gradient_x는 언제나 minimize.)

밑바닥부터 구현하기인 만큼, $\sum x^2$의 편미분값 $2x$를 손으로 입력해주지 말고 이전에 정의하였던 estimate_gradient함수를 통해서 찾아보자. (즉, $\frac{f(x+h)-f(h)}{h}$를 통해 **미분에 근사한** 식을 직접 구하는것!)


```python
v=[777*x for x in range(10)]#아무 숫자 v
[2*i for i in v]#gradient의 결과값은 이렇게 2*v로 나와야 할것이다!
```


    [0, 1554, 3108, 4662, 6216, 7770, 9324, 10878, 12432, 13986]
확인을 위해 아무 숫자를 가진 list(v)를  loss func$\sum x^2$에 넣어보자. 결과값은 위에 명시된 2*v가 나와야 한다.

```python
estimate_gradient(lambda x:sum([i*i for i in x]),v,h=1e-4)# x^2의 합이니까 gradient는 2x들 맞다
```


    [0.0,
     1554.0000796318054,
     3108.000159263611,
     4661.999940872192,
     6216.000020503998,
     7770.000100135803,
     9324.000179767609,
     10878.000259399414,
     12432.000041007996,
     13986.000120639801]

그럼 이제, 앞에꺼를 이어붙인 최종 완성본을 만들어보자, 이때 step은 어떻게 정하나? 다양한 방법이 있는데(e.g. 특정 step size로 고정한다, 몇번의 iter가 끝나면 점점 step size를 줄인다, 각 iter마다 다양한 step의 계산해서 가장 최적의 변화를 일으키는 step size를 택한다), 여기서는 각iter마다 여러번 step시도해보고 최적으로 간다.

이번에는 역시  loss func$\sum x^2$을 최적화할것이다. 우리는 이미 답이 $(0,0,..,0)$이 나와야 함을 알고 있다.


```python
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0 # set theta to initial value
    #target_fn = safe(target_fn) # safe version of target_fn
    value = target_fn(theta) # value we're minimizing
    cnt=0
    while True:
        cnt+=1
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                for step_size in step_sizes]
        
        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)#추정값중 가장 minimum
        if cnt%20==0:            
            print(next_theta,'this is next theta')
        next_value = target_fn(next_theta)
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value
```


```python
%%time
my_theta=[77*x for x in range(5)]
print(my_theta,' this is first theta')
finished_theta=minimize_batch(lambda x:sum([i*i for i in x]),lambda u:estimate_gradient(lambda x:sum([i*i for i in x]),u,h=1e-4),my_theta,tolerance=0.000001)
#람다항에서 u는 my_theta를 받는다
```

    [0, 77, 154, 231, 308]  this is first theta
    [-4.9423384496094513e-05, 0.887700135395562, 1.7754496933229689, 2.6631992519609184, 3.5509488109186123] this is next theta
    [-4.999335213775463e-05, 0.010185062218061793, 0.020420117778473162, 0.03065517334706549, 0.04089022891927558] this is next theta
    [-4.999981287931518e-05, 0.00023809138994608964, 0.0005261825924960112, 0.0008142737952761995, 0.0011023649981581664]  this is finished theta
    Wall time: 3.99 ms

역시나 0에 매우 가까운 값을 갖게 됨을 알 수 있다.


```python
# 지금 위에서 한것들은 summation 이 없는, 하나의 데이터에 대해서 한것이지만 원래는 mse같이 sum(error)로 해야한다.
# 즉 위의 예시의 경우 loss func는 단순히 [i*i for ]가 아니라 sum([i*i for] for range(len(data)))와 같은 형태로 되고
# 그에 맞춰 estimated gradient는 2*(sum(x))등으로 나와야 한다.
# 모든 데이터에 대해 loss func을 때리고 모든 데이터 기준으로 gradient를 구하고
#(이 경우, 혹은 많은 경우 loss func이 각 데이터에 대한 error의 sum이라 gradient도 각각의 sum으로 하게 된다)
# 최적화를 한다. 단순 sum인경우, 몇개(혹은 한개의 subset)의 데이터에 대해서만 구하는 sgd를 반복하는게 더 효율적인 경우가 많다
# (더 robust해지기도한다)
```

# 3. Stochastic Gradient Descent

지금까지 편의를 위해 $\sum_i^n x_i^2$과 같이 x들을 직접 설정해주었지만, 실제로 SSE등을 구할때는 $\sum_i^n(y_i-\hat{ y_i})^2$와 같이, 데이터에 대해서 직접 $(y_i-\hat{ y_i})^2$를구한후, 이들 모든 데이터에 대해 반복하고 평균을 내줘야 최종 cost function을 구할 수 있다 .

그러나, 만약 데이터가 1억개였다면, 이를 매 iteration마다 반복하는 것이 매우 버거운 일일 수 있다. 

그런데 cost function은, 사실 각각의 데이터에 대해 loss function을 구하고 이를 평균 취하여 구하는 것이다. 그렇다면, 어차피 선형 연산으로 +를 해주는 것이라면, 몇개의 데이터만 뽑아서 loss function을 구하고 update하는 것을 계속 반복하면 결국 전체 데이터에 대해서 update를 해주는것과 차이가 없지 않을까?

이러한 관점에서 등장한것이 stochastic gradient descent이다. 전체 데이터에 대해 평균을 내는 것이나, ramdom sampling 된 데이터에 대해 평균을 내는것이나, 확률적으로 그 **기댓값**은 같다는 이론에 기반을 둔 것이다. ($E(\frac{total\_data}{n_t})=E(\frac{sampled\_data}{n_s})$)

또한, 전체의 데이터에 대해서 적합을 하면 특정 local minima에 빠질 경우 절대 빠져나올 수가 없지만, 몇몇 샘플링 된 데이터에 대해서는 해당 local minima를 빠져나올 확률이 좀더 생긴다는 점에서도, local minima에 대해서 좀더 robust한 특성을 갖게되어 최적화방식에서 gradient descent를 대부분 대체하여 사용되는 방법이다.

딥러닝에서의 SGD는 보통 mini batch에 대해 업데이트를 하는 것을 의미하지만, 이때 sampling은 mini_batch를 뽑아서 하던, 임의의 10개를 뽑아서 하던, 1개를 뽑아서 하던 random sampling만을 만족시킨다면 모두 SGD의 범주에 포함된다고 할 수 있다. 여기에서는 random sampling된 1개의 데이터에 대해서만 gradient를 구한다.

이번에는 실제 데이터 전체에 대해 적합하는 만큼, improvement가 없더라도 100번의 iter를 더 돌면서 update를 해보고, 100연속 improvement가 없다면 멈추는 구문을 추가하였다.

```python
#여기서는 loss function에 y까지 필요하다
#위에서는 이미 '편차'의 제곱과 같이 input하나로 loss구햇으면 지금은 x,y둘다 넣어줘서 구하는형식!

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01,safe=False):
    

    data = zip(x, y)
    theta = theta_0                             # initial guess
    alpha = alpha_0                             # initial step size
    min_theta, min_value = None, float("inf")   # the minimum so far
    iterations_with_no_improvement = 0
    cnt_for_inf_loop=0
    # if we ever go 100 iterations with no improvement, stop
    while ((iterations_with_no_improvement < 100)&(cnt_for_inf_loop<1e10)):
        cnt_for_inf_loop+=1
        if safe:
            if cnt_for_inf_loop>1e5:
                print('too much iter!')
                break
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in zip(x, y) )

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            if cnt_for_inf_loop%20==1:
                print('min_theta updates',min_theta)
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            if  (iterations_with_no_improvement%20==5):
                print("iterations_with_no_improvement is growing...",iterations_with_no_improvement)
            alpha *= 0.9

        # and take a gradient step for each of the data points        
        """for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))"""
        indexes=[i for i in range(len(x))];random.shuffle(indexes)  
        for rand_i in indexes:
            gradient_i = gradient_fn(x[rand_i], y[rand_i], theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta

def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]
def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]
```

# 4. Gradient Descent로 Linear regression적합하기

이제 드디어, 모든 도구를 다 갖추었다. 맨 앞에서 했던 단순히 cov/var로써 적합하는 게 아닌, 데이터에 대해서 gradient를 구해서 최적해를 찾아보자.


```python
def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),       # alpha partial derivative
            -2 * error(alpha, beta, x_i, y_i) * x_i] # beta partial derivative
```

$(y_i-(\beta*x_1+\alpha))^2$가 목적함수니까 편미분은 각각

$\frac{\partial f}{\partial \alpha}=-2(\beta*x_1+\alpha)$

$$\frac{\partial f}{\partial \beta}=-2*x_1(\beta*x_1+\alpha)$$이 될것이다.


```python
%%time
random.seed(0)
theta = [random.random(), random.random()]
alpha, beta = minimize_stochastic(squared_error, 
                                  squared_error_gradient,
                                  num_friends_good,
                                  daily_minutes_good, 
                                  theta,
                                  0.0001)
print('Final parameter :', alpha,beta)
```

    min_theta updates [0.8444218515250481, 0.7579544029403025]
    min_theta updates [12.287728757179368, 1.6212965611920966]
    min_theta updates [15.560144551056357, 1.415040172643511]
    min_theta updates [17.822358779447857, 1.3909027828252298]
    min_theta updates [19.340607552944455, 1.1994438991099778]
    min_theta updates [20.387544972589787, 1.1035908826138416]
    min_theta updates [21.61801889250819, 1.001274091679216]
    min_theta updates [22.00488567039699, 0.9843192992048934]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 25
    iterations_with_no_improvement is growing... 45
    iterations_with_no_improvement is growing... 65
    iterations_with_no_improvement is growing... 85
    Final parameter : 22.93746417548679 0.9043371597664965
    Wall time: 546 ms

실제 정답값인 (22.95,0.903)에 매우 근접한 값을 찾을 수 있었다.

또한, loss function 기준 변수가 2개(알파와 베타), 즉 다변량이니까 위에서 짯던 partial gradient를 사용해볼수 있다. 

예를 들어, parameter가 각각 $\alpha=3, \beta=2$이었을때 데이터 $(x,y)=(6,18)$에 대해서 어떤 gradient를 갖는지 보자. 

```python
print('This is one with real derivative:',
      squared_error_gradient(6,18,(3,2)))#3*-2랑 3*-2*6
print('This is one with sudo derivative:',
      estimate_gradient(lambda coeff:squared_error(6,18,(coeff)),[3,2],h=1e-4))
```
```
This is one with real derivative: [-6, -36]
This is one with sudo derivative: [-5.9998999999777425, -35.99640000013338]
```

오예. 

따라서 마지막으로 이미 미분되어있는 식 말고, 이전에 짯던 partial derivative를 이용하여 gradient까지 직접 구하여 SGD를 해보자. 

이를 위해 특정 데이터 x_i,y_i를 받았을때 현재 coeff를 가지고 loss function인 squared_error에 대해서 gradient를 구할 수 있는 my_ols를 함수를 짜서 적합해본다. 물론, 조금 더 느릴수 있다.

```python
def my_ols(x_i,y_i,coeff):
    #print((x_i,y_i,coeff))
    return estimate_gradient(lambda coeff:squared_error(x_i,y_i,coeff),coeff,h=1e-4)    
```


```python
%%time
random.seed(0)
theta = [random.random(), random.random()]
alpha, beta = minimize_stochastic(squared_error, 
                                  my_ols,
                                  num_friends_good,
                                  daily_minutes_good, 
                                  theta,
                                  0.0001,safe=True)
print('Final parameter :', alpha,beta)
```

    min_theta updates [0.8444218515250481, 0.7579544029403025]
    min_theta updates [12.28805514597721, 1.6212228195868175]
    min_theta updates [15.560565522578992, 1.414956622720906]
    min_theta updates [17.822844346718213, 1.3908099012563253]
    min_theta updates [19.341138465457096, 1.199346739728466]
    min_theta updates [20.38810706840187, 1.1034939193107496]
    min_theta updates [21.618617382498233, 1.0011793142052967]
    min_theta updates [22.005494362904244, 0.9842221133970033]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    min_theta updates [22.862947253725665, 0.9141561880824236]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 25
    iterations_with_no_improvement is growing... 45
    iterations_with_no_improvement is growing... 65
    iterations_with_no_improvement is growing... 85
    Final parameter : 22.94909060432333 0.9041386511875767
    Wall time: 1.24 s

짠. 이번에도 역시 실제 정답값인 (22.95,0.903)에 매우 근접한 값을 찾을 수 있었다!

---

```python
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from functools import partial

def correlation(x, y):
    stdev_x = np.std(np.array(x))
    stdev_y = np.std(np.array(y))
    if stdev_x > 0 and stdev_y > 0:
        return np.cov([np.array(x),np.array(y)])[0][1] / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))    

def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def mean(x):
    return sum(x) / len(x)

```
