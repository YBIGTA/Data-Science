

```python
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from functools import partial
```

#### 아직 읽은 부분이 많지도 않아 충분히 바뀔 수 있습니다만, 한주동안 책을 접해보니, 다음과 같습니다. 

1. 가볍게 써져있다. 모델에 대한 근원적인 이야기나 수학적인 원리등은 거의(isl보다도 더) 안건들고 있습니다. 모델에 대한 간단한 특성설명, 파이썬으로 구현해보기가 메인입니다.


2. 그럼에도 불구하고 좋다. ISL로 모델에 대한 아주 얉은 지식이나마 쌓은 후에 접하기 좋은것 같다. 간단히 설명되는 특성에서도 나름 스스로 얻고 구글링해서 더 알아볼만한 점들이 많이 나온다. 무엇보다, numpy도 없이 모델을 구현하는데 나름 재미도 있다. (for문 돌리면서 멍때리고 쳐다보기..)


# Simple Linear Regression

$ y_i=\beta x_i + \alpha + \epsilon_i$


```python
#즉 이런식
def predict(alpha, beta, x_i):
    return beta * x_i + alpha
```

어떻게 최적의 계수 $\alpha, \beta$를 찾을까? 가장 대표적인 방법이 least square다. 


```python
def error(alpha, beta, x_i, y_i):
    """the error from predicting beta * x_i + alpha
    when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))
```

이를 통해 계산되는 계수는 다음과 같다. 식을 잘보면, 결국 $\beta$는 $\frac{cov(x,y)}{var(x)}$임을 알 수 있다. 

normal equation은 ISL에서 하도 많이 다뤘으니 넘어간다. 

$b_{0}=\bar y -b_{1}\bar x$

$b_{1}=\frac{\sum x_{i}(y_{i}-\bar y)}{\sum x_{i}(x_{i}-\bar x)}$


```python
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
```


```python
num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
print("alpha", alpha)
print("beta", beta)

```

    alpha 22.9167813834
    beta 0.908340529495
    

# Linear regression GD로 적합하기

GD는 우리가 알듯이 gradient를 구해서 loss func를 최소화하는 방향으로 parameter를 업데이트 해주는 최적화방식

<img src="./gradient.PNG">

아래와 같은 그림처럼, h->0에 수렴하는 아주 작은 값을 움직일때 값이 얼마나 변하는지를 의미한다.
<img src="./approximate.PNG">
그러나 파이썬에는 limit를 구현할 수 없기에, 여기서는 아주 작은 값을 input으로 근사를한다


```python
def sum_of_squares(v):
    """computes the sum of squared elements in v"""
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h
```


```python
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
```


```python
# 간단한 다변량함수정의
sum([x for x in range(10)])
```




    45



예를 들어, loss function을 $\sum^n_{i} i*{x_i}$ 이라고 정의해보면 다음과 같다


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




```python
#5번째에 대한 편미분
partial_difference_quotient(lambda x:my_sum(x)[1],[777*x for x in range(10)],5,1e-10)#1*i_1+2_i2....식으로 가니, i_5의 편미분은 5!
```




    4.94765117764473




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



이제 진짜 gd를 해보자. loss func이 $\sum x^2$라면 각각에 대한 편미분은 $2*x$일 것이다


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
#아쒸 귀찬아서 그냥 넘파이
def distance(v,w):
    return np.sum(np.power(np.array(v)-np.array(w),2))
```


```python
# pick a random starting point
v = [random.randint(-10,10) for i in range(3)]
v
#아항 x^2에서 음수이면 gradient도 음수니까, gradient방향으로 가는것(<-방향)이 언제나 maximize가 되네/반대방향은 minimize
```




    [7, 0, 4]




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



요전에 짜놓은 근사gradient함수로도 돌려보자


```python
v=[777*x for x in range(10)]
[2*i for i in v]
```




    [0, 1554, 3108, 4662, 6216, 7770, 9324, 10878, 12432, 13986]




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




```python
%%time
# 내 예시
v=[777*x for x in range(10)]
tolerance = 0.00000000000000000001
stop_cnt=0
while True:
    gradient=estimate_gradient(lambda x:sum([i*i for i in x]),v,h=1e-4)
    #gradient = sum_of_squares_gradient(v) # compute the gradient at v
    next_v = step(v, gradient, -0.01) # take a negative gradient step
    if distance(next_v, v) < tolerance: # stop if we're converging
        break
    v = next_v # continue if we're not
    stop_cnt+=1
    if stop_cnt>1e5:
        print('break!')
        break
print(v)
```

    [-4.9999999999976834e-05, -4.9999707218096565e-05, -4.999941443621704e-05, -4.999912165432856e-05, -4.9998828872450376e-05, -4.999853609056292e-05, -4.99982433086865e-05, -4.9997950526805814e-05, -4.999765774492273e-05, -4.999736496304008e-05]
    CPU times: user 68 ms, sys: 4 ms, total: 72 ms
    Wall time: 70.6 ms
    


```python
%%time
# 내 예시
v=[777*x for x in range(10)]
tolerance = 0.00000000000000000001
stop_cnt=0
while True:
    gradient=estimate_gradient(lambda x:np.sum(np.power(np.array(x)-3,2)),v,h=1e-4)
    #gradient = sum_of_squares_gradient(v) # compute the gradient at v
    next_v = step(v, gradient, -0.01) # take a negative gradient step
    if distance(next_v, v) < tolerance: # stop if we're converging
        break
    v = next_v # continue if we're not
    stop_cnt+=1
    if stop_cnt>1e5:
        break
print(v)
```

    [2.9999499999988686, 2.9999500002916517, 2.9999500005844326, 2.9999500008772157, 2.9999500011699967, 2.9999500014627789, 2.9999500017555607, 2.9999500020483429, 2.9999500023411247, 2.999950002633907]
    CPU times: user 300 ms, sys: 0 ns, total: 300 ms
    Wall time: 297 ms
    

앞에꺼를 이어붙인 최종 완성본, 이때 step은 어떻게 정하나? 다양한 방법이 있는데, 여기서는 그냥 각iter마다 여러번 step시도해보고 최적으로 갓다


```python
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0 # set theta to initial value
    #target_fn = safe(target_fn) # safe version of target_fn
    value = target_fn(theta) # value we're minimizing
    
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                for step_size in step_sizes]
        
        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)#추정값중 가장 minimum
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
    [-1.0011717677116394e-05, 61.599990013521165, 123.19998998055235, 184.7999900057912, 246.39999003103003] this is next theta
    [-1.8000719137489796e-05, 49.27998203015886, 98.55998198827729, 147.83998200460337, 197.11998202092946] this is next theta
    [-2.4389009922742844e-05, 39.42397562309634, 78.84797559154686, 118.27197560365312, 157.69597561575938] this is next theta
    [-2.9511284083127975e-05, 31.539170503267087, 63.07837047416251, 94.61757048871368, 126.1567704959889] this is next theta
    [-3.3607648219913244e-05, 25.23132640158292, 50.46268638171023, 75.69404639094137, 100.92540639653453] this is next theta
    [-3.688910510390997e-05, 20.1850511173252, 40.370139101927634, 60.55522710835794, 80.74031511478825] this is next theta
    [-3.951026883441955e-05, 16.14803089319321, 32.296101282103336, 48.44417168556538, 64.5922420926654] this is next theta
    [-4.1608473111409694e-05, 12.918414715160907, 25.836871026062, 38.75532734787703, 51.67378367423953] this is next theta
    [-4.328740033088252e-05, 10.334721771643672, 20.669486819773738, 31.00425187699875, 41.33901693877124] this is next theta
    [-4.462981451069936e-05, 8.26776741746653, 16.53557945610737, 24.803391501569422, 33.07120355066945] this is next theta
    [-4.570392775349319e-05, 6.614203933851968, 13.22845356481048, 19.84270320122596, 26.456952839915175] this is next theta
    [-4.6563172872993164e-05, 5.2913531469603186, 10.582752851863916, 15.874152560860239, 21.1655522721303] this is next theta
    [-4.7250523493858054e-05, 4.233072517537948, 8.466192281506665, 12.699312048772299, 16.93243181762955] this is next theta
    [-4.780042672791751e-05, 3.3864480140000524, 6.772943825243601, 10.159439639101947, 13.545935454210849] this is next theta
    [-4.8240337946481304e-05, 2.70914841121521, 5.418345060210413, 8.127541711308822, 10.83673836337357] this is next theta
    [-4.8592283974358e-05, 2.167308728958915, 4.334666048155441, 6.5020233690572695, 8.669380690698063] this is next theta
    [-4.887380100626615e-05, 1.7338369831936689, 3.467722838539885, 5.201608695250343, 6.935494552557657] this is next theta
    [-4.90990430535021e-05, 1.3870595865359974, 2.7741682708218605, 4.1612769561877485, 5.5483856420510165] this is next theta
    [-4.927923669129086e-05, 1.1096376692325975, 2.2193246166573886, 3.3290115649490417, 4.438698513645704] this is next theta
    [-4.9423384496094513e-05, 0.887700135395562, 1.7754496933229689, 2.6631992519609184, 3.5509488109186123] this is next theta
    [-4.9538709134822057e-05, 0.7101501083131438, 1.4203497546568542, 2.130549401565446, 2.8407490487298332] this is next theta
    [-4.963096777998999e-05, 0.5681100866504067, 1.1362698037250283, 1.7044295212508445, 2.272589238982718] this is next theta
    [-4.970477363031023e-05, 0.45447806932057233, 0.9090058429795675, 1.3635336170009396, 1.8180613911857364] this is next theta
    [-4.976381973165189e-05, 0.363572455455639, 0.7271946743828437, 1.090816893601371, 1.454439112948684] this is next theta
    [-4.9811055724546804e-05, 0.2908479643642252, 0.5817457395061751, 0.8726435148808278, 1.1635412903585092] this is next theta
    [-4.9848845051769786e-05, 0.2326683714910942, 0.4653865916047515, 0.6981048119044821, 0.9308230322863693] this is next theta
    [-4.987907598064112e-05, 0.18612469719285585, 0.37229927328361256, 0.5584738495235833, 0.7446484258288351] this is next theta
    [-4.990326085696495e-05, 0.14888975775422075, 0.2978294186269679, 0.44676907961895296, 0.5957087406631185] this is next theta
    [-4.992260871361509e-05, 0.11910180620337929, 0.2382535349015633, 0.3574052636951155, 0.4765569925305231] this is next theta
    [-4.9938086943424054e-05, 0.09527144496272832, 0.19059282792125076, 0.28591421095610103, 0.38123559402442453] this is next theta
    [-4.9950469538373454e-05, 0.07620715597021865, 0.15246426233703403, 0.22872136876485616, 0.3049784752195457] this is next theta
    [-4.996037564208855e-05, 0.06095572477617206, 0.12196140986964399, 0.18296709501188246, 0.24397278017565927] this is next theta
    [-4.996830051118284e-05, 0.048754579820940336, 0.09755912789572085, 0.14636367600949796, 0.19516822414053903] this is next theta
    [-4.9974640412009386e-05, 0.038993663856749405, 0.0780373023165657, 0.117080940807607, 0.15612457931242896] this is next theta
    [-4.997971232711951e-05, 0.031184931085393885, 0.06241984185324989, 0.09365475264608591, 0.12488966344994784] this is next theta
    [-4.99837698661465e-05, 0.024937944868312245, 0.049925873482600025, 0.07491380211686488, 0.09990173075995601] this is next theta
    [-4.9987015894592535e-05, 0.019940355894651096, 0.03993069878608013, 0.05992104169348944, 0.07991138460796254] this is next theta
    [-4.9989612716655474e-05, 0.015942284715720442, 0.03193455902886422, 0.04792683335479117, 0.06391910768636985] this is next theta
    [-4.9991690172918046e-05, 0.012743827772576266, 0.025537647223091313, 0.038331466683832555, 0.05112528614909535] this is next theta
    [-4.999335213775463e-05, 0.010185062218061793, 0.020420117778473162, 0.03065517334706549, 0.04089022891927558] this is next theta
    [-4.99946817096239e-05, 0.008138049774449867, 0.016326094222778816, 0.024514138677652876, 0.032702183135420455] this is next theta
    [-4.999574536789994e-05, 0.006500439819559806, 0.013050875378222991, 0.01960131094212244, 0.02615174650833601] this is next theta
    [-4.999659629443404e-05, 0.005190351855647583, 0.010430700302578418, 0.015671048753697743, 0.020911397206668667] this is next theta
    [-4.999727703548784e-05, 0.004142281484518066, 0.00833456024206278, 0.012526839002958225, 0.0167191177653349] this is next theta
    [-4.999782162850436e-05, 0.003303825187614408, 0.0066576481936501414, 0.010011471202366567, 0.013365294212267804] this is next theta
    [-4.9998257302787466e-05, 0.0026330601500915147, 0.005316118554920106, 0.007999176961893251, 0.010682235369814256] this is next theta
    [-4.999860584223564e-05, 0.0020964481200731783, 0.004242894843936088, 0.006389341569514588, 0.008535788295851401] this is next theta
    [-4.9998884673799595e-05, 0.0016671584960585416, 0.0033843158751488576, 0.005101473255611668, 0.006818630636681139] this is next theta
    [-4.999910773904534e-05, 0.0013237267968468378, 0.0026974527001190868, 0.004071178604489337, 0.005444904509344913] this is next theta
    [-4.999928619123516e-05, 0.001048981437477468, 0.002147962160095266, 0.0032469428835914674, 0.0043459236074759305] this is next theta
    [-4.9999428952987016e-05, 0.0008291851499819666, 0.0017083697280762147, 0.0025875543068731716, 0.0034667388859807337] this is next theta
    [-4.9999543162389855e-05, 0.0006533481199855716, 0.001356695782460967, 0.002060043445498535, 0.002763391108784587] this is next theta
    [-4.9999634529912805e-05, 0.0005126784959884573, 0.0010753566259687737, 0.001638034756398829, 0.0022007128870276697] this is next theta
    [-4.999970762392981e-05, 0.0004001427967907655, 0.0008502853007750176, 0.0013004278051190622, 0.0017505703096221358] this is next theta
    [-4.999976609914409e-05, 0.0003101142374326121, 0.0006702282406200141, 0.0010303422440952497, 0.0013904562476977082] this is next theta
    [-4.999981287931518e-05, 0.00023809138994608964, 0.0005261825924960112, 0.0008142737952761995, 0.0011023649981581664] this is next theta
    [-4.999985030345213e-05, 0.00018047311195687184, 0.00041094607399680904, 0.0006414190362209595, 0.0008718919985265329] this is next theta
    CPU times: user 8 ms, sys: 0 ns, total: 8 ms
    Wall time: 4.66 ms
    


```python
finished_theta
```




    [-4.999981287931518e-05,
     0.00023809138994608964,
     0.0005261825924960112,
     0.0008142737952761995,
     0.0011023649981581664]




```python
# 지금 위에서 한것들은 summation 이 없는, 하나의 데이터에 대해서 한것이지만 원래는 mse같이 sum(error)로 해야한다.
# 즉 위의 예시의 경우 loss func는 단순히 [i*i for ]가 아니라 sum([i*i for] for range(len(data)))와 같은 형태로 되고
# 그에 맞춰 estimated gradient는 2*(sum(x))등으로 나와야 한다.
# 모든 데이터에 대해 loss func을 때리고 모든 데이터 기준으로 gradient를 구하고
#(이 경우, 혹은 많은 경우 loss func이 각 데이터에 대한 error의 sum이라 gradient도 각각의 sum으로 하게 된다)
# 최적화를 한다. 단순 sum인경우, 몇개(혹은 한개의 subset)의 데이터에 대해서만 구하는 sgd를 반복하는게 더 효율적인 경우가 많다
# (더 robust해지기도한다)
```

# SGD


```python
20%10
```




    0




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
```

# Linear regression with GD


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

$(y_i-(\beta*x_1+\alpha))^2$가 목적함수니까 
편미분은 각각

$\frac{\partial f}{\partial \alpha}=-2(\beta*x_1+\alpha)$

$\frac{\partial f}{\partial \beta}=-2*x_1(\beta*x_1+\alpha)$


```python
num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
```


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
print(alpha,beta)
```

    min_theta updates [0.8444218515250481, 0.7579544029403025]
    min_theta updates [4.701380978660914, 2.3833796204983435]
    min_theta updates [10.217899664766477, 1.7835548138327117]
    min_theta updates [12.287728757179368, 1.6212965611920966]
    min_theta updates [14.047502022406492, 1.5803213515647367]
    min_theta updates [15.560144551056357, 1.415040172643511]
    min_theta updates [16.781583851859274, 1.3876659110352516]
    min_theta updates [17.822358779447857, 1.3909027828252298]
    min_theta updates [18.63023462871273, 1.173555309820907]
    min_theta updates [19.340607552944455, 1.1994438991099778]
    min_theta updates [19.930394583654962, 1.2032175399709748]
    min_theta updates [20.387544972589787, 1.1035908826138416]
    min_theta updates [20.778105520617107, 1.0812551534779034]
    min_theta updates [21.61801889250819, 1.001274091679216]
    min_theta updates [22.00488567039699, 0.9843192992048934]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    min_theta updates [22.629905119097057, 0.9271109796421689]
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
    min_theta updates [22.87685393638785, 0.9135697020404603]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 25
    iterations_with_no_improvement is growing... 35
    iterations_with_no_improvement is growing... 45
    iterations_with_no_improvement is growing... 55
    iterations_with_no_improvement is growing... 65
    iterations_with_no_improvement is growing... 75
    iterations_with_no_improvement is growing... 85
    iterations_with_no_improvement is growing... 95
    22.93746417548679 0.9043371597664965
    CPU times: user 364 ms, sys: 8 ms, total: 372 ms
    Wall time: 366 ms
    

막간을 이용하여, loss function 기준 변수가 2개(알파와 베타)이니까 위에서 짯던 partial gradient를 사용해볼수 있다. 값이 거의 또같게나온다!

마지막으로 이미 미분되어있는 식 말고, 이전에 짯던 partial derivative를 이용하여 SGD를 해보자. 물론, 조금 더 느릴수 있다


```python

def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) # add h to just the ith element of v
        for j, v_j in enumerate(v)]
    # [원래a,원래b,,,,i+h,...]이런식의 리스트 반환, 즉 i만 h만큼 아주 조금 옴겻을때!
    return (f(w) - f(v)) / h

print(partial_difference_quotient(lambda coeff:squared_error(6,18,(coeff)),[3,2],0,1e-4))

print(estimate_gradient(lambda coeff:squared_error(6,18,(coeff)),[3,2],h=1e-4))
```

    -5.9998999999777425
    [-5.9998999999777425, -35.99640000013338]
    


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
print(alpha,beta)
```

    min_theta updates [0.8444218515250481, 0.7579544029403025]
    min_theta updates [4.701484600764317, 2.3833210872057395]
    min_theta updates [10.21816509336554, 1.7834858321200304]
    min_theta updates [12.288055145976347, 1.6212228195862766]
    min_theta updates [14.047879855185016, 1.5802415282423554]
    min_theta updates [15.560565522577724, 1.4149566227203518]
    min_theta updates [16.782040443364565, 1.3875779957486778]
    min_theta updates [17.822844346719286, 1.3908099012565491]
    min_theta updates [18.630745495074656, 1.1734681851217665]
    min_theta updates [19.341138465455973, 1.1993467397282471]
    min_theta updates [19.930942771544895, 1.203126862452319]
    min_theta updates [20.388107068406708, 1.10349391930982]
    min_theta updates [20.7786792123857, 1.081159763696011]
    min_theta updates [21.61861738250418, 1.0011793142046834]
    min_theta updates [22.00549436290897, 0.9842221133964275]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    min_theta updates [22.630531921423643, 0.927012616817173]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    min_theta updates [22.862947253726773, 0.9141561880824074]
    iterations_with_no_improvement is growing... 5
    min_theta updates [22.878676334077685, 0.9131503913167371]
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 5
    iterations_with_no_improvement is growing... 15
    iterations_with_no_improvement is growing... 25
    iterations_with_no_improvement is growing... 35
    iterations_with_no_improvement is growing... 45
    iterations_with_no_improvement is growing... 55
    iterations_with_no_improvement is growing... 65
    iterations_with_no_improvement is growing... 75
    iterations_with_no_improvement is growing... 85
    iterations_with_no_improvement is growing... 95
    22.9490906043242 0.9041386511874705
    CPU times: user 920 ms, sys: 4 ms, total: 924 ms
    Wall time: 916 ms
    

# 막간 알쓸신잡


```python
def my_least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta"""
    beta = (np.corrcoef([x, y])[0][1]) * np.std(y) / np.std(x);print(beta,'beta1')
    beta=(np.cov([x,y])[0][1])/(np.var(x));print(beta,'beta2')
    alpha = np.mean(y) - beta * np.mean(x)
    return alpha, beta

print(my_least_squares_fit(num_friends_good,daily_minutes_good))
#아이쒸....왜달라....
```

    0.903865945606 beta1
    0.908340529495 beta2
    (22.916781383374218, 0.90834052949500987)
    


```python
x=num_friends_good
y=daily_minutes_good
```


```python
print(np.std(y))
print(np.power(np.sum(np.power(y-np.mean(y),2))/len(y),0.5),'same!')#np.std는 /n이다 (population method)
```

    9.84366818785
    9.84366818785 same!
    


```python
print(np.var(y))
print(np.sum(np.power(y-np.mean(y),2))/(len(y)),'same!')#np.var도 /n이다 (population method)
```

    96.8978033925
    96.8978033925 same!
    


```python
print(np.cov(y))
print(np.sum(np.power(y-np.mean(y),2))/(len(y)),'different!')#np.cov는 /(n-1)이다!! (sampling method)
print(np.cov(y,ddof=0))#np.cov는 denominator=len(input)-ddof를 사용하는데 cov의 경우 defalt로 ddof=1이다.
```

    97.3774954884651
    96.8978033925 different!
    96.8978033924628
    


```python
print(np.corrcoef(x,y)[0][1])
print(np.cov(x,y,ddof=0)[0][1]/(np.std(x)*np.std(y)),'same,when ddof=1!')
#그래서 corr의 정의대로 cov/(std1*std2)로 하려면 ddof=0을따로 입력해줘야 한다. 왜이렇게 해놧을까...
```

    0.573679211567
    0.573679211567 same,when ddof=1!
    
