{:toc}

> (가독성과 재생산성을 모두 살리기 위해 맨 아래부분에 직접사용한 함수들을 모아놓았습니다. 코드를 실행하려면 맨아래 cell의 함수를 먼저 실행하고 위에서 부터 순서대로 실행하면 됩니다.)

원래 multiple regression을 다루면서 ridge등을 구현할 예정이었으나 이번 공모전이 classification문제라서, 진도를 유동적으로 바꿨습니다.



총 200명의 data가 있고, 각 직장 경력, 연봉을 통해서 해당 사람이 premium계정을 구입했는지를 classificatino하는 문제가 있다고 해보자.

```python
import numpy as np
import matplotlib.pyplot as plt

data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
```


```python
print(len(data))
print(data[0]) #years of experience at work, salary, whether use premium account
```

```python
200
(0.7, 48000, 1)#데이터 예시. 0.7년 근무에 연봉 48000, premium=True
```


우선, 데이터를 처리할 수 있도록 x,y형태로 나눠준다. 그리고 학습을 빠르게 할 수 있도록 normalize도 해준다. 


```python
x = [[1] + list(row[:2]) for row in data] # each element is [1, experience, salary], note '1' is for intercept term
y = [row[2] for row in data]        # each element is paid_account
rescaled_x = rescale(x)# 평균 0, 분산1로 만들어주기
```


```python
X_mean=np.mean((np.array(x)),axis=0).repeat(np.array(x).shape[0]).reshape(np.array(x).T.shape).T
X_std=np.std((np.array(x)),axis=0).repeat(np.array(x).shape[0]).reshape(np.array(x).T.shape).T
scaled=(x-X_mean)/X_std
scaled[np.isnan(scaled)]=np.array(x)[np.isnan(scaled)]
#numpy사용해서 scaling한것과 값같다.
```

첫번째 시도로 0과1의 클래스를 나타내는 categorical 변수에 단순히 multiple regression을 이용하여 적합해보자. 이런 경우에도, **값은 나온다**(이런 잘못된 사용을 지양하기 위해 모델에 대한 이해가 있어야 한다)

다음은 적합된 값들과 실제 값들을 plot으로 그린것이다.


```python
from sklearn.linear_model import LinearRegression#지난번에 구현한 GD로 적합할수도 잇으나 logistic에 집중하기 위해 생략하고 sklearn import.
try: del linear_reg;#just for sure
except:pass
linear_reg=LinearRegression().fit(x,y)
y_hat=linear_reg.predict(x)
plt.xlabel('y_hat')
plt.ylabel('y(true_value)')
plt.plot(y_hat,y,'o');plt.show()
```


![output_7_0](https://user-images.githubusercontent.com/31824102/44199229-0d19d180-a133-11e8-994b-739ca0a59cd2.png)


값을 보면 다음과 같은 특징이 있다.
당근, 0과 1의 값만으로 나오지 않고, **-0.5~1.50**정도의 continuous 한 값으로 나왓다. 우리가 원하는 0과 1로 구분하는 task와 거리가 있음을 알 수 있다.

그렇다면, linear regression이 class가 1인 애들은 1에 가까운 값을 내도록 적합하려 했을것이기에,
continuous한 값을 각각 '1에 가까울 확률'이라고 보려고 한다.
그러나 이경우에도 0보다 작은 값들, 1보다 큰값들도 잇어 바로 확률로 해석하기가 힘들다.

굳이 1.5의 값을 유지하며 linear regression을 고수하려고 해보자.(이경우 error를 0.5로 설정할것이다.) 그럼 다음과 같은 가정의 위반을 직면하게 된다.

우선, X~Y가 선형이 될 수 없다. (Y는 [0,1]의 값을 가져야 하기에, boundary가 있다.) (**선형성 가정 위반.**)

두번째로, 각각의 값들에 대해 [0,1]의 boundary를 맞춰주기 위해 x가 커지거나 작아지거나 할경우 error term도 그만큼 커져야 함을 의미하게 되는데, 이는 error가 x의 수준에 의존하지 않고 언제나 iid라는 linear regression의 가정을 위반하게 된다. 구체적으로는 분산도 달라지고(**등분산 가정 위반**), **평균**도 달라지게 된다(**mean0가정 위반**)

또한, Y변수가 0,1과 같은 discrete한 값들이기에, Y와 적합값Y_hat의 차인 error역시 normal분포를 띈다고 가정하기 힘들다(**오차의 정규가정 위반**)

무엇보다, 이중 가장 중요한것은 mean0위반이다. 위의 regression을 고수한다면 큰 x값에 대해 error도 mean0가 아니고, 결국 우리의 추정이 **biased된 값**이라는 결론을 내게 된다.(i.e. $E(y.hat)\neq Y$)

좀더 이해를 수우러하게 하기 위해 단변수에 대해 적합을 해서 그래프로 확인해보자. 다음은 premium~salary의 관계를 보기 위해 산점도를 그려본 것이다. (왠지 모르게 이 예제 데이터에서는 salary가 적은 사람들일 수록 premium이다)


```python
plt.plot(np.array(x)[:,2],y,'o')
plt.xlabel('salary')
plt.ylabel('whether premium');plt.show()
```


![output_10_0](https://user-images.githubusercontent.com/31824102/44199230-0d19d180-a133-11e8-9556-0bca187471b2.png)

이에 대해 linear regression을 적합하여, 0과 1을 예측한다고 해보자. 적합된 회귀선을 함께 그림으로 그려보면, 다음과 같다.

```python
try: del linear_reg;
except:pass
linear_reg=LinearRegression().fit(np.array(x)[:,2].reshape(-1,1),y)
coeff=linear_reg.coef_
intercept=linear_reg.intercept_
plt.plot(np.array(x)[:,2],y,'o');
plt.plot(np.linspace(2e4,12e4,1000),np.linspace(2e4,12e4,1000)*coeff+intercept)
plt.xlabel('salary')
plt.ylabel('whether premium')
plt.show()
```


![output_11_0](https://user-images.githubusercontent.com/31824102/44199231-0d19d180-a133-11e8-8d2d-5f53bed1cfda.png)


그림으로 보면 알 수 있듯이, salary가 큰사람일 수록 y_hat값은 무한정 음수로 떨어질 것이고, (error=y-y_hat이므로) error는 0(또는 간혹1)을 유지하기 위해 positive 쪽으로 무한정 커지게 될것이다. 즉, 등분산가정이 깨질뿐더러 mean0가정이 깨지고, error의 mean0가정이 깨진다는 것은 E(y_hat)=y가 될 수 없다는 것이므로 우리 모델이 bias되어있다는 것!

# then, how?

앞서 말했듯이, linear regression이 class가 1인 애들은 1에 가까운 값을 내도록 적합하려 했을것이기에,
continuous한 값을 각각 '1에 가까울 확률'이라고 보려고 하는 것은 합리적이다. 그러나 [-inf,inf]가 나올 수 있는 이 적합값y_hat을 **어떻게 [0,1]이라는 probabilty의 틀**에 넣어 해석할 수 있을지에 대한 문제로 귀결된다.

이를 boundary problem이라고 한다. 로지스틱 회귀에서는 이 문제를, 결과값에 특정한 함수를 씌워서 해결한다

# Logistic function

로지스틱 회귀에서는 이 문제를, 결과값에 특정한 함수 로지스틱을 씌워서 해결한다.

로지스틱 함수는 다음과 같다. 

$$
f(x)=\frac{1}{1+e^{-x}}
$$


```python
def logistic(x):
    try:
        return 1.0 / (1 + math.exp(-x))
    except OverflowError:
        return 1e-9
#원래 try구문만으로 완벽한 식이지만, x에 지나치게 큰수가 들어갈 경우 error가 나서 except구문을 만들어줬다
#뒤에서 gd할때 step size를 이상한거로(e.g. 100) 막 넣어서 계산하면 e^5000이런게 나와버려서 계산을 못한다
```

함수에 대한 설명을 하기 전에 우선 그림으로 직관적으로 봐보자. [-10,10]이었던 X값들이 logistic함수이라는 특정한를 씌우니 [0,1]로 수렴하는 것을 볼 수 있다.


```python
y=[]
for x in np.linspace(-10,10,100):
    y.append(logistic(x))
plt.plot(np.linspace(-10,10,100),np.linspace(-10,10,100));plt.show()
plt.plot(np.linspace(-10,10,100),y)
#plt.ylim(-10,10)
plt.axhline(y=0.5,alpha=0.5,color='y');plt.axvline(x=0,alpha=0.5,color='y');
plt.show()
```

![output_17_0](https://user-images.githubusercontent.com/31824102/44199234-0db26800-a133-11e8-840a-43f265ac0b92.png)

![output_17_0](https://user-images.githubusercontent.com/31824102/44199235-0e4afe80-a133-11e8-95c8-58787cbfbc9a.png)


x가 커질수록, $e^{-x}$는 0에 가까워져서 $\frac{1}{1+e^{-x}}$는 1에 가까워지고,

x가 작아질수록, $e^{-x}$는 inf에 가까워져서 $\frac{1}{1+e^{-x}}$는 0에 가까워지며,

x가 0일때 $e^{-x}$는 (모든 수가 그러하듯이) 1이므로 $\frac{1}{1+e^{-x}}=1/2$가 된다

이렇게 logistic function을 결과값에 씌워 [0,1]의 값을 갖게 하여, 해당 데이터가 이러한 feature들을 갖고 있을때 y=True가 나올 **conditional probability**로 표현할 수 있도록 한다.

>  참고로 logistic function은 딥러닝의 non-linear function에서 sigmoid함수라고도 불린다. 엄밀히 말하지면 sigmoid function은 S자 모양의 모든 함수를 지칭하는 것이고, 로지스틱은 그중 한가지 예시일 뿐이다 [참고](https://en.wikipedia.org/wiki/Sigmoid_function)

> 또하나의 참고로 sigmoid func의 미분은 다음과 같게 된다(back prop에 사용된다)
>
> $f(x)=\frac{1}{1+e^{-x}}=\frac{1}{g(x)}$, where $g(x)=1+e^{-x}$
>
> $$f'(x)=\frac{d}{dx}({\frac{1}{g(x)}})=\frac{d}{dx}(g(x)^{-1})=-1*g(x)^{-2}*g'(x)=-\frac{g'(x)}{g(x)^{2}}$$
>
> $g'(x)=\frac{d}{dx}(1+e^{-x})=-1*e^{-x}$근데 이는 다시 g(x)의 형태로 (굳이)나타낼 수 있다. (그럼 최종 결과가 f(x)의 형태로 나와서 합성함수 계산에서 편하게 표현할수 있다)
>
> $-1*e^{-x}=-e^{-x}=1-(1+e^{-x})=1-g(x), $ 
>
> $\therefore g'(x)=1-g(x)$
>
> $\therefore f'(x)=-\frac{1-g(x)}{g(x)^2}=\frac{\frac{-1}{g(x)}+1}{g(x)}=\frac{1}{g(x)}*(1-\frac{1}{g(x)})$
>
> $\therefore f'(x)=f(x)*(1-f(x))$
>
> 이를 코드로 표현해서 도함수 역시 정의해놓는다


```python
def logistic_prime(x):
    return logistic(x) * (1 - logistic(x))
```

우리가 하는것은 앞에 나왔던 regression의 값을 0,1으로 만들어줘서 확률로써 해석하는 것이다.
수식으로 표현하면 다음과 같다

$y_i=f(\boldsymbol X_i \beta)+\epsilon_i$, 여기서 f는 로지스틱함수이다

로지스틱 함수는 least square방식이 아니라 maximum likelihood로써 적합을 한다(구체적으로는, mse는 loss function이 convex형태로 표현되지 않는다고 한다)

> 또한, 그렇기에 logistic regression의 coefficient는 likelihood ratio test를 이용하여 검정한다. 이쯤가면 너무 통계통계한 부분이므로 여기서 생략하겠다.(사실 잘 모른다.)
>
> 또한 그로 인해 기존의 여러 가정들이 필요하지 않게 되었지만, 반면에 mse보다 수렴속도가 더 느리다고 한다. [참고](http://www.statisticssolutions.com/wp-content/uploads/wp-post-to-pdf-enhanced-cache/1/assumptions-of-logistic-regression.pdf)

여기서는 likelihood를 maximize하기 위해 지난시간에 사용했던 GD 함수를 이용하여 적합을 해보겠다.

# likelihood
likelihood는 단순하게 해당 y들을 맞출수 있을 가장 그럴듯한 beta라고 보면 된다.

고정된 beta에서, 특정 y_i에 대해서 우리 모델이 뱉는 확률(i.e 모델이 추정한 bernouli 분포는)은 다음과 같이 말할 수 있다.(yi=1 or 0의 두가지 경우이다)

$p(y_i\vert\boldsymbol X_i,\beta)=f(\boldsymbol x_i \beta)^{y_i}*(1-f(\boldsymbol x_i \beta))^{1-y_i}$

이를 단순히 beta에 대해 표현하면, beta에 대한 likelihood가 되어 해당 likelihood를 maximize하는 beta를 구할 수 있다. 또한, 이는 계산상의 이유로(underflow방지, derivative에서 훨씬 수월) log를 씌워 log_likelihood로 표현하기도 한다. (log는 단조증가여서 정보의 손실이 없다)

![likelihood_logistic](https://user-images.githubusercontent.com/31824102/44199314-44887e00-a133-11e8-8093-fac2130062ad.PNG)


```python
def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        #print('when True,',logistic(dot(x_i, beta)))
        return math.log(logistic(dot(x_i, beta)))
    else:
        #print(x_i,beta)
        #print('dot is ',dot(x_i,beta))
        #print('here is ValueError',logistic(dot(x_i, beta)))
        return math.log(1 - logistic(dot(x_i, beta))+1e-9)
#likelihood에서 결국 boundary된 확률값들을 cummprod해주는게 필요한데(아님 log_cumsum) 
#근데 해당 확률이 0에 계산오차로 완전수렴해 버리는 경우도 잇어서(물론 그경우 mle의 고려대상이 아니겟지만, for문돌며 gd할때 오류가 나버려서 1e-9를 해주었다)
```

또한, 만약 우리가 (대부분의 ML 기법이 기본가정으로 깔고가듯이) **각 데이터가 서로 독립이라고 가정**한다면, 
위에서 구한 likelihood를 모든 data y_i들에 대해서 cumproduct를 해서 전체 데이터에 대한 likelihood와, 그 전체 데이터의 likelihood를 maximize하는 beta를 구할 수 잇을 것이다.(혹은 log-likelihood의 경우. 이 경우 cumprod가 아니라 cumsum이라는 것만 달라진다)


```python
def logistic_log_likelihood(x, y, beta):#log의 경우 cumprod가 cumsum이 되므로, 단순히 더해준다.
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
        for x_i, y_i in zip(x, y))
```

> 데이터들간의 독립이 뭘까? 예를 들자면 키~몸무게로 regression을할때 데이터안에 4명의 가족이 끼어있다면, 그들은 indep가 아니다(error항이 서로 비슷할 것이다)
 은 timeseries데이터들(사람1의 10살키,몸무게..사람1의 80살키,몸무게.사람n의 80살키,몸무게)를 예측할때에도 같은 timeseries는 서로 연관이 되어있다고 봐야한다

즉, 모든 데이터$y_i$에 대해 구한 우리의 최종적인 목적함수 log likelihood는 다음과 같다. 

(bold체는 하려하다가 귀찬아서 생략했지만, $\boldsymbol \beta$는 모두 $(\beta_0+...+\beta_p)$의 차원을 가진 벡터이고 $x_i$ 역시 intercept를 포함해서 $(1+x_1+..+x_p)$의 벡터이다)

$J(\boldsymbol\beta)=\log L(\beta)=\frac{1}{n}[\sum_i^n y_i\log f(x_i\beta)+(n-\sum y_i)\log(1-f(x_i\beta))$]

해당 최종식을 gradient descent를 통해 최적화를 할것이다. 그러기 위해선 각 $\beta_j$에 대해 gradient를 구해야 한다.

수학시간은 아니지만, 해당 식의 gradient를 이번만 짚고 넘어가보자.

> $$\begin{align}\frac{\partial J(\boldsymbol\beta)}{\partial\beta_j }&=\frac{\partial}{\partial\beta_j}[\frac{1}{n}\sum_i^n y_i\log f(x_i\beta)+(n-\sum y_i)\log(1-f(x_i\beta))]\\ &= \frac{1}{n}\sum [\frac{y_i\frac{\partial}{\partial\beta_j}f(x_i\beta)}{f(x_i\beta)}+\frac{(1-y_i)\frac{\partial}{\partial \beta_j}(1-f(x_i\beta))}{1-f(x_i\beta)}] \\&=\frac{1}{n}\sum [\frac{y_i\ f'(x_i\beta)*x_{ij}}{f(x_i\beta)}+\frac{(1-y_i)(-f'(x_i\beta))*x_{ij}}{1-f(x_i\beta)}],\\& \because \frac{\partial x_i\beta}{\partial\beta_j}= \frac{\partial (x_{i1}\beta_1+..+x_{ij}\beta_j+..)}{\partial\beta_j}=x_{ij}\\ & =\frac{1}{n}\sum [\frac{y_i*f(x_i\beta)*(1-f(x_i\beta))*x_{ij}}{f(x_i\beta)}-\frac{(1-y_i)*(f(x_i\beta)*(1-f(x_i\beta))*x_{ij}}{1-f(x_i\beta)}],\\&(\because f'(A)=[f(A)*(1-f(A)])\\&=\frac{1}{n}\sum [{y_i*(1-f(x_i\beta))*x_{ij}}-{(1-y_i)*f(x_i\beta)*x_{ij}}]\\&=\frac{1}{n}\sum[(y-f(x_i\beta))x_{ij}]\end{align}$$
>
> 즉, 특정 데이터 $x_i$에 대해 계수 $\beta_j$의 gradient는 '$\frac{1}{n}\sum[(y-f(x_i\beta))x_{ij}]$'이다.

다음은 앞서 정리한 내용을 바탕으로 구현된 logistic regression이다.대해 모든 데이터(혹은 sgd의 경우 샘플링된 데이터)에 대해  log likelihood의 편미분값을 더하고(혹은 평균내고. $\frac{1}{n}$은 상수이므로 최적화에 영향을 미치지 않는다) 그로써 learning rate(여기선 step size라고 notation이 써져있다)를 정하여 그만큼 update를 한다.


```python
def logistic_log_likelihood(x, y, beta):#log의 경우 cumprod가 cumsum이 되므로, 단순히 더해준다.
    """for x_i, y_i in zip(x, y):
        print(logistic_log_likelihood_i(x_i, y_i, beta))"""
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
        for x_i, y_i in zip(x, y))

def logistic_log_partial_ij(x_i, y_i, beta, j):
    """here i is the index of the data point,
    j the index of the derivative"""

    return (y_i - logistic(dot(x_i, beta))) * x_i[j]
    
def logistic_log_gradient_i(x_i, y_i, beta):#각 계수 1~j에 대해 구한 partial deriv를 가진 1*j 벡터
    """the gradient of the log likelihood 
    corresponding to the i-th data point"""
    #print('this is x_i',x_i)
    #print([logistic_log_partial_ij(x_i, y_i, beta, j)for j, _ in enumerate(beta)])
    return [logistic_log_partial_ij(x_i, y_i, beta, j)for j, _ in enumerate(beta)]
            
def logistic_log_gradient(x, y, beta):#그 벡터들을 모든 데이터 i개에 대해서 더해준다
    #print('this si array',np.array([logistic_log_gradient_i(x_i, y_i, beta)for x_i, y_i in zip(x,y)]))
    return np.sum(np.array([logistic_log_gradient_i(x_i, y_i, beta)for x_i, y_i in zip(x,y)]),axis=0)
```


```python
%%time
print ("logistic regression:")

random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

# want to maximize log likelihood on the training data
fn = partial(logistic_log_likelihood, x_train, y_train)
gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# pick a random starting point
beta_0 = [1, 1, 1]

# and maximize using gradient descent
beta_hat = maximize_batch(fn, gradient_fn, beta_0)#모든 데이터에 대해 gradient를 구해서 sum을하고, 그만큼 update하는것.

print("beta_batch", beta_hat)
```

    logistic regression:
    next_theta [-0.98255588458629939, 2.3852843058873878, -2.1113132057851853]
    next_theta [-1.6231205844767986, 2.9325005457297024, -2.9904220026290869]
    next_theta [-1.6845316575352558, 3.4541986134410352, -3.2120994183075102]
    next_theta [-1.763457666332795, 3.5988394252854343, -3.4983715931343213]
    next_theta [-1.8178162645708138, 3.7684258957250201, -3.6508389962038663]
    next_theta [-1.8506352010086142, 3.8733776113222493, -3.7382869413690938]
    next_theta [-1.8711133955049506, 3.9396845394768709, -3.7907247408212608]
    next_theta [-1.884057747511656, 3.9818799446732993, -3.8231967753842135]
    next_theta [-1.8923209208041496, 4.0087980588515881, -3.8437741842667696]
    next_theta [-1.8976291568201162, 4.0259916573174905, -3.8570150985780529]
    next_theta [-1.9010484929931222, 4.0369915442924063, -3.8656049709518974]
    next_theta [-1.9032580841621607, 4.0440371450804316, -3.8712051897327044]
    next_theta [-1.9049000320852507, 4.0502654786007559, -3.8743316201458753]
    next_theta [-1.9057510521365857, 4.0524794633637535, -3.8770025027550772]
    beta_batch [-1.906182482651773, 4.0530838693737428, -3.8788953691426906]
    Wall time: 1.04 s

```python
%%time
beta_0 = [1, 1, 1]
beta_hat = maximize_stochastic(logistic_log_likelihood_i,
                           logistic_log_gradient_i,
                           x_train, y_train, beta_0)
print("beta stochastic", beta_hat)
```

    min_theta updates [1, 1, 1]
    min_theta updates [-1.2679993666825518, 2.0110699795905385, -1.930051041435664]
    min_theta updates [-1.4768306698812361, 2.743237461070941, -2.6392424401722647]
    min_theta updates [-1.6057675528751156, 3.1559211370221583, -3.0229551097263743]
    min_theta updates [-1.690510565123893, 3.415967989691299, -3.2665060117900717]
    min_theta updates [-1.7504435935307276, 3.583775137243253, -3.4405470946821524]
    min_theta updates [-1.793646263969424, 3.704489929737186, -3.5590487749898037]
    min_theta updates [-1.8174922511616782, 3.7925518487145067, -3.6454846681199076]
    min_theta updates [-1.8408885825791077, 3.8624878817342068, -3.70068299132356]
    min_theta updates [-1.8547808871820977, 3.9134815135640904, -3.743702967792522]
    min_theta updates [-1.8662758232017913, 3.9481631856179966, -3.777549032672308]
    iterations_with_no_improvement is growing... 5
    min_theta updates [-1.8914001807751846, 3.9918356822733148, -3.820470489514673]
    iterations_with_no_improvement is growing... 5
    min_theta updates [-1.8888930102866952, 4.008210432054739, -3.834111274491687]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    min_theta updates [-1.8993159351402242, 4.0229180503190705, -3.85260331633106]
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
    beta stochastic [-1.901794592305049, 4.0402458845427285, -3.8670118939700515]
    Wall time: 1.12 s

둘다 실제값인 beta_hat = [-1.90, 4.05, -3.87]과 거의 동일한 예측값을 추정하였음을 알 수 있다.

덤으로 test set에 대해 precision과 recall을 계산해 보았다.


```python
#decision boundary를 0.5로 설정해서 precision과 recall을 봐본다. 근데 unbalanced인경우 등 threshold를 바꿔줄수도

true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(x_test, y_test):
    predict = logistic(dot(beta_hat, x_i))

    if y_i == 1 and predict >= 0.5:  # TP: paid and we predict paid
        true_positives += 1
    elif y_i == 1:                   # FN: paid and we predict unpaid
        false_negatives += 1
    elif predict >= 0.5:             # FP: unpaid and we predict paid
        false_positives += 1
    else:                            # TN: unpaid and we predict unpaid
        true_negatives += 1

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print("precision", precision)
print("recall", recall)
```

    precision 0.9333333333333333
    recall 0.8235294117647058

```python
predictions = [logistic(dot(beta_hat, x_i)) for x_i in x_test]
plt.scatter(predictions, y_test)
plt.xlabel("predicted probability")
plt.ylabel("actual outcome")
plt.title("Logistic Regression Predicted vs. Actual")
plt.show()
```


![output_38_0](https://user-images.githubusercontent.com/31824102/44199236-0e4afe80-a133-11e8-8ca5-77c130fe2eec.png)

### 덧. 해석에 관하여

그럼 왜 굳이 하고 많은 S자형 함수 중에 logistic function이 classification에 사용된것일까? 많은 해석을 할 수 있겠지만 그중 한가지 이유는 바로 logistic function을 사용할 경우 모델을 odds의 개념으로 해석할 수 있기 때문이다. 

### 오즈(Odds)란?

오즈란 간단히 말해 **(일어날확률)/(일어나지 않을확률)**이다. 통계학에서, 혹은 도박에서 승산의 상대적인 강도를 토현할때 사용되는 지표라고 보면 된다. 예를들어 도박에서 이길 확률($P(X=True)$)이 0.75, 질 확률($1-P(X=True)$)이 0.25이면, 이때 odds=$\frac{P(X)}{1-P(X)}=\frac{0.75}{0.25}=3$이라고 볼 수 있다. 우리가 평소에 이길확률이 질확률 보다 3배 높은 게임이라고 표현하는 것이 오즈를 이용한 표현이다. 왜 굳이 확률을 확률로써 보면 되지 오즈를 새로 정의하냐? 로지스틱함수와 같은 특정상황에서는 해석을할때 확률로써하는 것보다 **오즈로써 해석하는것이 훨씬 수월한 상황들이 있기 때문**(아래 좀더 설명한다).

다시 돌아와서, binary classification을 하는 로지스틱함수는, 사실 해당 feature를 가진 데이터가 $X=True$를 가질 확률을 나타내는것이라 볼 수 있다. 

> X:사건이 일어났는지 안났는지. 사건이 일어났으면 X=1, 안일어 났으면 X=0
>
> $$E(Y)=1*P(X)+0*(1-P(X))=P(X)$$

그리고 여기서, 우리가 로지스틱함수를 사용하였기때문에, 오즈에 로그를 씌운 **로짓을 X에 대해 선형으로써 표현**할 수 있게 된다.

> $Y=\frac{1}{1+e^{-f(x)}}, $ where $f(x)=\beta_0+\beta_1x_1+..+\beta_px_p$
>
> $Y=\frac{e^{f(x)}}{e^{f(x)}+1}$
>
> $e^{f(x)}*Y+Y=e^{f(x)}$
>
> $e^{f(x)}(Y-1)=-Y$
>
> $e^{f(x)}=\frac{Y}{1-Y}$
>
> $\therefore f(x)=\log{\frac{Y}{1-Y}}$
>
> $E(f(x))=\log(\frac{E(Y)}{1-E(Y)})=\log(\frac{P(X)}{1-P(X)})$
>
> $\therefore f(x)=\log(\frac{P(X)}{1-P(X)})+\epsilon=logit+\epsilon$

즉 로지스틱 회귀는 해당 Y변수의 로짓(log_odds)가 X변수들과 선형관계가 있다고 가정을 하는것이라고 해석할 수 있다.

### 그에 따른 계수에 해석

또한, 이렇게 로짓으로 표현가능하다는 점에서, 계수에 대한 해석 역시 선형회귀처럼 용이해진다. linear regression에서 계수$\beta_j$는 다른 변수($x_i$)들이 고정되어있는 상태에서 $x_j$가 1증가했을때 **Y의 증가량의 기대값**(즉, $\beta_j$=3이라면 $x_j$가 1 증가하면 $E(Y)$는 3증가한다. )을 의미한다. 반면에 logistic regression에서 Y는 '해당 데이터가 class1에 속할 확률'을 나타내는데, 확률은 S자 개형을 가지고 있기때문에 X의 증가량이 우리모델에서 **확률(Y)자체에 어떤 영향을 미치는지는 상수로 표현할 수 없다**. (sigmoid의 개형을 생각해보자. x가 1증가할때 y가 몇증가할지는 각 구간마다 다 다르다.)

그러나 로지스틱함수로서 로짓의 개념을 사용할 수 있기에, 로지스틱회귀에서 계수$\beta_j$는 $x_j$가 1증가했을때 Y의 **로짓**의 증가량의 기대값이라는 의미를 갖게 된다. 

> 위의 증명에서 아주 간단하게 이어진다.
>
> $f(x)=\beta_0+\beta_1x_1+..+\beta_px_p=\log(\frac{P(X)}{1-P(X)})+\epsilon$,
>
> 따라서 $x_j$의 계수가 $\beta_j$라면, '$x_j$가 1 증가할때(i.e. $\beta_j(x_j+1)$) 로짓이 $\beta_j$만큼 증가할것(add)'이라고 표현할 수 있다.
>
> 로짓이 아니라 오즈로 표현한다면, 다음과 같이 지수부분에 들어가서 product로 표현되므로, 
>
>  '$x_j$가 1 증가할때 **오즈비가** $e^{\beta_j}$배 만큼 커질것(product)'라고도 표현할 수 있다. 사실상 둘이 같은 말이니 헷갈리지 말자.(나는 헷갈려서 구글링 1시간을 소비하고 정리하였다ㅋㅋ)
>
> $e^{-f(x)}=\frac{P(X)}{1-P(X)}=odds_1$ => $$e^{-f(x)+\beta_j}=[\frac{P(X)}{1-P(X)}]*e^{\beta_j}=odds_1*e^{\beta_j}=odds_2$$
>
> $\therefore odds_2=odds_1*e^{\beta_j}$

끝! 아래는 정리와 사용한 함수들. 맨아래cell을 실행하고 위에를 실행하면 모두 제대로 실행됩니답

---

참고: 

가장 정리 잘해놓은곳. 로짓과 선형관계 증명:http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/)

전반적인 강의안 : http://www.columbia.edu/~so33/SusDev/Lecture_10.pdf

gradient 편미분 증명:https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated
로지스틱에 대한 andew ng강의정리:http://gnujoow.github.io/ml/2016/01/29/ML3-Logistic-Regression/)
mle에 대하여:https://onlinecourses.science.psu.edu/stat414/node/191/)

```python
from functools import partial
def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
        for j, v_j in enumerate(v)]
    # [원래a,원래b,,,,i+h,...]이런식의 리스트 반환, 즉 i만 h만큼 아주 조금 옴겻을때!
    return (f(w) - f(v)) / h

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
            if cnt_for_inf_loop%10==1:
                print('min_theta updates',min_theta)
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            if  (iterations_with_no_improvement%10==5):
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

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0 # set theta to initial value
    #target_fn = safe(target_fn) # safe version of target_fn
    value = target_fn(theta) # value we're minimizing
    #print('value:',value)
    
    while True:
        #print('theta',theta)
        gradient = gradient_fn(theta)
        #print('gradient',gradient)
        next_thetas = [step(theta, gradient, -step_size)
                for step_size in step_sizes]
        #print('text_thetas',next_thetas)
        
        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)#추정값중 가장 minimum
        print('next_theta',next_theta)
        
        next_value = target_fn(next_theta)
        #print('next_value',next_value)
        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def distance(v,w):
    return np.sum(np.power(np.array(v)-np.array(w),2))

def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
        for v_i, direction_i in zip(v, direction)]
def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]
#다변량 square일때 각각의 편미분들

#다변량일때의 gradient는 편미분을해서 구한다.

def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
        for j, v_j in enumerate(v)]
    # [원래a,원래b,,,,i+h,...]이런식의 리스트 반환, 즉 i만 h만큼 아주 조금 옴겻을때!
    return (f(w) - f(v)) / h #i만 아주 약간의 차이가 있는 두 리스트를 각각의 f(x)에 넣엇을때의 차이(를 h로 나누기)

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h)
        for i, _ in enumerate(v)]
#여러 변수의 각각의 편미분이 나오는 리스트 반환

def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

def least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

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

def error(alpha, beta, x_i, y_i):
    """the error from predicting beta * x_i + alpha
    when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

#즉 이런식
def predict(alpha, beta, x_i):
    return beta * x_i + alpha

import math

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of features of first row
    return num_rows, num_cols

def get_column(A, j):
    return [A_i[j] # jth element of row A_i
        for A_i in A] # for each row A_i
def standard_deviation(x):
    return math.sqrt(variance(x))
def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)
def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i
               for v_i, w_i in zip(v, w))
def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def scale(data_matrix):
    """returns the means and standard deviations of each column"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix,j))
        for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix,j))
        for j in range(num_cols)]
    return means, stdevs
def mean(x):
    return sum(x) / len(x)

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) # given i, create a list
        for j in range(num_cols)] # [entry_fn(i, 0), ... ]
        for i in range(num_rows)] # create one list for each i


def rescale(data_matrix):
    """rescales the input data so that each column
    has mean 0 and standard deviation 1
    leaves alone columns with no deviation"""
    means, stdevs = scale(data_matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)

def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error,
                               squared_error_gradient,
                               x, y,
                               beta_initial,
                               0.001)

def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y) # pair corresponding values
    train, test = split_data(data, 1 - test_pct) # split the data set of pairs
    x_train, y_train = zip(*train) # magical un-zip trick
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
        negate_all(gradient_fn),
        x, y, theta_0, alpha_0)
def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)
def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]
```
