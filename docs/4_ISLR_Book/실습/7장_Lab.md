R Notebook
================

Lab: Non-linear Modeling
========================

Wage 데이터를 다시 살펴보며, 7장에서 나온 예시 그림들이 어떻게 나왔는지 살펴본다.

``` r
library(ISLR)
library(ggplot2)
attach(Wage)
```

7.8.1 Polynomial Regressio and Step Functions
---------------------------------------------

그림 7.1이 어떻게 나왔는지 살펴보자.

``` r
fit = lm(wage~poly(age,4),data=Wage)
summary(fit)
```

    ## 
    ## Call:
    ## lm(formula = wage ~ poly(age, 4), data = Wage)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -98.707 -24.626  -4.993  15.217 203.693 
    ## 
    ## Coefficients:
    ##                Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)    111.7036     0.7287 153.283  < 2e-16 ***
    ## poly(age, 4)1  447.0679    39.9148  11.201  < 2e-16 ***
    ## poly(age, 4)2 -478.3158    39.9148 -11.983  < 2e-16 ***
    ## poly(age, 4)3  125.5217    39.9148   3.145  0.00168 ** 
    ## poly(age, 4)4  -77.9112    39.9148  -1.952  0.05104 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 39.91 on 2995 degrees of freedom
    ## Multiple R-squared:  0.08626,    Adjusted R-squared:  0.08504 
    ## F-statistic: 70.69 on 4 and 2995 DF,  p-value: < 2.2e-16

``` r
coef(summary(fit))
```

    ##                 Estimate Std. Error    t value     Pr(>|t|)
    ## (Intercept)    111.70361  0.7287409 153.283015 0.000000e+00
    ## poly(age, 4)1  447.06785 39.9147851  11.200558 1.484604e-28
    ## poly(age, 4)2 -478.31581 39.9147851 -11.983424 2.355831e-32
    ## poly(age, 4)3  125.52169 39.9147851   3.144742 1.678622e-03
    ## poly(age, 4)4  -77.91118 39.9147851  -1.951938 5.103865e-02

또한 age, age^2, age^3, age^4를 직접적으로 얻기 위해 poly()을 사용할 수도 있다.

``` r
fit2 = lm(wage~poly(age,4,raw=T),data=Wage)
coef(summary(fit2))
```

    ##                             Estimate   Std. Error   t value     Pr(>|t|)
    ## (Intercept)            -1.841542e+02 6.004038e+01 -3.067172 0.0021802539
    ## poly(age, 4, raw = T)1  2.124552e+01 5.886748e+00  3.609042 0.0003123618
    ## poly(age, 4, raw = T)2 -5.638593e-01 2.061083e-01 -2.735743 0.0062606446
    ## poly(age, 4, raw = T)3  6.810688e-03 3.065931e-03  2.221409 0.0263977518
    ## poly(age, 4, raw = T)4 -3.203830e-05 1.641359e-05 -1.951938 0.0510386498

또 다른 방법도 있다.

``` r
fit2a = lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
coef(fit2a)
```

    ##   (Intercept)           age      I(age^2)      I(age^3)      I(age^4) 
    ## -1.841542e+02  2.124552e+01 -5.638593e-01  6.810688e-03 -3.203830e-05

이제 우리가 예측을 원하는 age의 값을 정하고 predict()을 사용해보자.

``` r
agelims = range(age)
age.grid = seq(from=agelims[1],to=agelims[2])
preds = predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands = cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)
```

데이터 산점도를 찍고 4차 다항 적합을 그려보자!

``` r
par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0)) 
# mar: margin to be specified on the four sides of the plot(vector shape: bottom, left, top, right)
# oma: size of the outer margins in lines of text(vector shape: bottom, left, top, right)
plot(age,wage,xlim=agelims,cex=0.5,col='darkgrey')
title("Degree-4 polynomial",outer=T)
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-6-1.png)

다항 회귀를 할 때, 몇 차까지 적합을 해야하는지 결정해야하는데 가설 검정이 그 방법 중 하나이다. 1차부터 5차까지 적합한 모델들에 대해 anova를 실시한다. 귀무가설은 모델 M1이 충분한 정보를 담고 있다는 것이고 대립가설은 더 복잡한 M2 모델이 필요하다는 것이다. 이러한 anova를 실기하기 위해서는 M1의 예측 변수들이 M2의 subset이어야만 한다.

``` r
fit.1 = lm(wage~age,data=Wage)
fit.2 = lm(wage~poly(age,2),data=Wage)
fit.3 = lm(wage~poly(age,3),data=Wage)
fit.4 = lm(wage~poly(age,4),data=Wage)
fit.5 = lm(wage~poly(age,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: wage ~ age
    ## Model 2: wage ~ poly(age, 2)
    ## Model 3: wage ~ poly(age, 3)
    ## Model 4: wage ~ poly(age, 4)
    ## Model 5: wage ~ poly(age, 5)
    ##   Res.Df     RSS Df Sum of Sq        F    Pr(>F)    
    ## 1   2998 5022216                                    
    ## 2   2997 4793430  1    228786 143.5931 < 2.2e-16 ***
    ## 3   2996 4777674  1     15756   9.8888  0.001679 ** 
    ## 4   2995 4771604  1      6070   3.8098  0.051046 .  
    ## 5   2994 4770322  1      1283   0.8050  0.369682    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

살펴보면, fit.1 vs fit.2에서 p-value가 작게 나왔으므로 귀무가설 기각! quadratic이 simple보다 좋다는 뜻이고 fit.2 vs fit.3을 보면 p-value가 작으므로 fit.3(cubic)이 좋다는 뜻이고 fit.3 vs fit.4을 보면 p-value가 0.5 근처이므로 나름 fit.4(polynomial degree 4)가 좋다는 뜻! 종합하면 3차와 4차가 나름 좋다는 결과이다.

anova 테스트는 이렇게 차수만 다른 모델뿐만 아니라 여러 모델들을 비교하는데도 쓰일 수 있다.

``` r
fit.1 = lm(wage~education+age,data=Wage)
fit.2 = lm(wage~education+poly(age,2),data=Wage)
fit.3 = lm(wage~education+poly(age,3),data=Wage)
anova(fit.1,fit.2,fit.3)
```

    ## Analysis of Variance Table
    ## 
    ## Model 1: wage ~ education + age
    ## Model 2: wage ~ education + poly(age, 2)
    ## Model 3: wage ~ education + poly(age, 3)
    ##   Res.Df     RSS Df Sum of Sq        F Pr(>F)    
    ## 1   2994 3867992                                 
    ## 2   2993 3725395  1    142597 114.6969 <2e-16 ***
    ## 3   2992 3719809  1      5587   4.4936 0.0341 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

물론 cross-validation도 쓰일 수 있다.

이제 일년에 $250000 이상을 쓰는지 분류하는 문제를 생각해보자. 전거 거비 비슷하고 family="binomial"로 바꿔주면 된다.

``` r
fit.logistic = glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
preds=predict(fit.logistic,newdata=list(age=age.grid),se=T)
```

하지만 여기서 구한 표준편차는 logit에 대한 것이므로 이를 변환해서 Pr(Y=1 | X)에 대한 표준편차로 바꿔야 한다.

``` r
tr.fit = exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit + 2*preds$se.fit , preds$fit - 2*preds$se.fit)
tr.se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
tr.se.bands = as.data.frame(tr.se.bands)
names(tr.se.bands) = c('up','down')
tr.se.bands$age = age.grid
```

이것은 predict에서 type="link"가 기본값이기 때문에 생기는 현상인데, type="response"로 해주면 바로 나온다! 하지만 양끝쪽에서 음수의 확률이거나 1넘어갈 있으므로 주의해야한다!! 차라리 안쓰는게 나을 듯.

``` r
Wage$over250 = I(wage>250)
fit.mat = as.data.frame(cbind(age.grid,tr.fit))
names(fit.mat) = c('age','fit')
Wage$over250[which(Wage$over250)] = 0.2
ggplot(Wage,aes(x=age,y=over250)) + geom_point(shape='|',size=3,fill="darkgrey") + scale_y_continuous(limits=c(0,0.2)) + 
  geom_line(data=fit.mat,aes(x=age,y=fit),size=2) + geom_line(data=tr.se.bands,aes(x=age,y=up),linetype="dashed") +
  geom_line(data=tr.se.bands,aes(x=age,y=down),linetype="dashed") + 
  ggtitle("fitting logistic polynomial regression with CI")
```

    ## Warning: Removed 9 rows containing missing values (geom_path).

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-11-1.png)

step function을 적용하기 위해서 cut() function을 사용한다.

``` r
table(cut(age,4))
```

    ## 
    ## (17.9,33.5]   (33.5,49]   (49,64.5] (64.5,80.1] 
    ##         750        1399         779          72

``` r
fit.step = lm(wage~cut(age,4),data=Wage)
coef(summary(fit.step))
```

    ##                         Estimate Std. Error   t value     Pr(>|t|)
    ## (Intercept)            94.158392   1.476069 63.789970 0.000000e+00
    ## cut(age, 4)(33.5,49]   24.053491   1.829431 13.148074 1.982315e-38
    ## cut(age, 4)(49,64.5]   23.664559   2.067958 11.443444 1.040750e-29
    ## cut(age, 4)(64.5,80.1]  7.640592   4.987424  1.531972 1.256350e-01

``` r
pred.step = predict(fit.step,newdata=list(age=age.grid),se=T)
step.se.bands = as.data.frame(cbind(pred.step$fit + 2*pred.step$se.fit, pred.step$fit - 2*pred.step$se.fit, pred.step$fit, age.grid))
names(step.se.bands) = c('up','down','fit','age')

ggplot(Wage,aes(x=age,y=wage)) + geom_point(color="darkgray") + geom_line(data=step.se.bands,aes(x=age,y=fit),size=1) + 
  geom_line(data=step.se.bands,aes(x=age,y=up),linetype="dashed") + geom_line(data=step.se.bands,aes(x=age,y=down),linetype="dashed") + 
  ggtitle("fitting step functions with dashed CI")
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-12-1.png)

7.8.2 Splines
-------------

regression splines을 하기 위해서 splines library을 불러온다. bs() function은 knots와 함께 basis function을 만들어주는데 기본값으로 cubic splines가 생성된다.

``` r
library(splines)
fit.spl = lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
pred.spl=predict(fit.spl,newdata=list(age=age.grid),se=T)
sple.se.bands = as.data.frame(cbind(pred.spl$fit + 2*pred.spl$se.fit,
      pred.spl$fit - 2*pred.spl$se.fit,
      pred.spl$fit,
      age.grid))
names(sple.se.bands) = c('up','down','fit','age')
ggplot(Wage,aes(x=age,y=wage)) + geom_point(color="darkgray") + geom_line(data=sple.se.bands,aes(x=age,y=fit),size=1) + 
  geom_line(data=sple.se.bands,aes(x=age,y=up),linetype="dashed") + geom_line(data=sple.se.bands,aes(x=age,y=down),linetype="dashed") + 
  ggtitle("fitting splines regression with dashed CI")
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-13-1.png)

여기서 우리는 25, 40, 60이라는 knots를 설정했지만는 bs()에서 df option을 이용해서 사분위수를 이용하여 knots를 만들 수도 있다. 자유도를 6으로 한다는 것은 knots을 6-3 = 3개로 한다는 뜻이다. 또한 bs()에서느 degree을 통해서 차수를 조정할 수 있으며, 기본값은 degree=3이므로 cubic spline을 해준다.

``` r
attr(bs(age,df=6),"knots")
```

    ##   25%   50%   75% 
    ## 33.75 42.00 51.00

regression splines이 아닌 natural spline을 적합하고 싶으면 ns() function을 사용하면 된다. (natural splines이 무엇인지는 274쪽 참조.) degree가 10인 regression splines와 비교.

``` r
fit.ns = lm(wage~ns(age,df=4),data=Wage)
pred.ns = predict(fit.ns,newdata=list(age=age.grid),se=T)
ns.se.bands = as.data.frame(cbind(pred.ns$fit + 2*pred.ns$se.fit,
      pred.ns$fit - 2*pred.ns$se.fit,
      pred.ns$fit,
      age.grid))
names(ns.se.bands) = c('up','down','fit','age')

fit.bs.10 = lm(wage~bs(age,degree=10),data=Wage)
pred.bs.10 = predict(fit.bs.10,newdata=list(age=age.grid),se=T)
bs.se.bands = as.data.frame(cbind(pred.bs.10$fit,age.grid))
names(bs.se.bands) = c('fit','age')

ggplot(Wage,aes(x=age,y=wage)) + geom_point(color="darkgray") + geom_line(data=ns.se.bands,aes(x=age,y=fit),size=1) + 
  geom_line(data=ns.se.bands,aes(x=age,y=up),linetype="dashed") + geom_line(data=ns.se.bands,aes(x=age,y=down),linetype="dashed") + 
   geom_line(data=bs.se.bands,aes(x=age,y=fit),color='red')
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-15-1.png)

좀더 구불부굴한 10차 regression splines을 볼 수 있다. smoothing spline을 적합하기 위해서는 smooth.spline()을 사용한다.

``` r
fit.smooth = smooth.spline(age,wage,df=16)
fit.smooth2 = smooth.spline(age,wage,cv=TRUE)
```

    ## Warning in smooth.spline(age, wage, cv = TRUE): cross-validation with non-
    ## unique 'x' values seems doubtful

``` r
fit.smooth$lambda # df=16으로 지정했을 때, 적절한 람다값 출력
```

    ## [1] 0.0006537868

``` r
fit.smooth2$lambda # cv를 통해 선택한 smoothness에 대한 람다 값
```

    ## [1] 0.02792303

``` r
fit.smooth2$df #cv를 통해 선택한 smoothness에 대한 자유도 값.
```

    ## [1] 6.794596

local regression을 하기 위해서는 loess()을 사용한다. 또는 locfit library을 사용할 수도 있다.

``` r
fit.local1 = loess(wage~age,span=0.2,data=Wage)
fit.local2 = loess(wage~age,span=0.4,data=Wage)
plot(Wage$age,Wage$wage,xlim=agelims,cex=.5,col='darkgrey')
lines(age.grid,predict(fit.local1,data.frame(age=age.grid)),col='red',lwd=2)
lines(age.grid,predict(fit.local2,data.frame(age=age.grid)),col='blue',lwd=2)
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-17-1.png)

span이 작은 빨간색 선이 더 구불구불한 것을 볼 수 있다.

7.8.3 GAMs
----------

year와 age는 natural splines으로, education은 범주형 변수로 취급하여 GAMs을 해보자. 어차피 basis func에 따른 linear regression이기 때문에 lm 함수를 사용한다.

``` r
gam1 = lm(wage~ns(year,4)+ns(age,5)+education,data=Wage)
gam1
```

    ## 
    ## Call:
    ## lm(formula = wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)
    ## 
    ## Coefficients:
    ##                 (Intercept)                 ns(year, 4)1  
    ##                      46.949                        8.625  
    ##                ns(year, 4)2                 ns(year, 4)3  
    ##                       3.762                        8.127  
    ##                ns(year, 4)4                  ns(age, 5)1  
    ##                       6.806                       45.170  
    ##                 ns(age, 5)2                  ns(age, 5)3  
    ##                      38.450                       34.239  
    ##                 ns(age, 5)4                  ns(age, 5)5  
    ##                      48.678                        6.557  
    ##         education2. HS Grad     education3. Some College  
    ##                      10.983                       23.473  
    ##    education4. College Grad  education5. Advanced Degree  
    ##                      38.314                       62.554

smoothing splines나 다른 일반적인 애들을 적용하고 싶으면 gam library을 사용하면 된다. gam library의 s() func은 smoothing spline으로 적합하고 싶을 때 사용한다.

``` r
library(gam)
```

    ## Loading required package: foreach

    ## Loaded gam 1.16

``` r
gam.m3 = gam(wage~s(year,4)+s(age,5)+education,data=Wage)
par(mfrow=c(1,3))
plot(gam.m3,se=TRUE,col="blue")
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-19-1.png)

살펴보면 year에 대한 함수는 거의 선형이다. anova test을 통해서 어떤 모델이 가장 좋은 것인지 알아볼 수 있다. gam.m1: year을 뺀 모델 gam.m2: year에 대해 linear func을 적용한 모델 gam.m3: year에 대해 spline func을 적용한 모델

``` r
gam.m1 = gam(wage~s(age,5)+education,data=Wage)
gam.m2 = gam(wage~ year+s(age,5)+education,data=Wage)
anova(gam.m1,gam.m2,gam.m3)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: wage ~ s(age, 5) + education
    ## Model 2: wage ~ year + s(age, 5) + education
    ## Model 3: wage ~ s(year, 4) + s(age, 5) + education
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1      2990    3711731                          
    ## 2      2989    3693842  1  17889.2 0.0001419 ***
    ## 3      2986    3689770  3   4071.1 0.3483897    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

gam.m2 vs gam.m3에서 (세 번째 결과) p-value가 높게 나온 것으로 봐서, linear func을 적용한 것이 더 좋음을 알 수 있다. 또한 gam.m1 vs gam.m2에서 (두 번째 결과) p-value가 낮게 나온 것으로 봐서, year을 아예 빼는 것 보다는 linear func을 넣는 것이 좋음을 알 수 있다.

``` r
summary(gam.m3)
```

    ## 
    ## Call: gam(formula = wage ~ s(year, 4) + s(age, 5) + education, data = Wage)
    ## Deviance Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -119.43  -19.70   -3.33   14.17  213.48 
    ## 
    ## (Dispersion Parameter for gaussian family taken to be 1235.69)
    ## 
    ##     Null Deviance: 5222086 on 2999 degrees of freedom
    ## Residual Deviance: 3689770 on 2986 degrees of freedom
    ## AIC: 29887.75 
    ## 
    ## Number of Local Scoring Iterations: 2 
    ## 
    ## Anova for Parametric Effects
    ##              Df  Sum Sq Mean Sq F value    Pr(>F)    
    ## s(year, 4)    1   27162   27162  21.981 2.877e-06 ***
    ## s(age, 5)     1  195338  195338 158.081 < 2.2e-16 ***
    ## education     4 1069726  267432 216.423 < 2.2e-16 ***
    ## Residuals  2986 3689770    1236                      
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Anova for Nonparametric Effects
    ##             Npar Df Npar F  Pr(F)    
    ## (Intercept)                          
    ## s(year, 4)        3  1.086 0.3537    
    ## s(age, 5)         4 32.380 <2e-16 ***
    ## education                            
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

여기서 p-values는 귀무가설: linear relationship 대립가설: non-linear relationship에서의 p-value을 나타내며 year에 대한 p-value가 큰 것은 귀무가설 채택 = linear relationship이라는 뜻이고 age에 대해서는 non-linear relationship = 대립가설 채택이므로 non-linear func을 적용하는 것이 합당해 보인다!

training data에 대해서 예측을 해보자.

``` r
pred.gam = predict(gam.m2,newdata=Wage)
```

또한 local regression을 building blocks을 lo() func을 이용해서 사용할 수 있다.

``` r
gam.lo=gam(wage~year + lo(age,span=0.7)+education,data=Wage)
plot(gam.lo,se=TRUE,col="green")
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-23-1.png)![](7장_Lab_files/figure-markdown_github/unnamed-chunk-23-2.png)![](7장_Lab_files/figure-markdown_github/unnamed-chunk-23-3.png)

또한 lo() func으로 interaction term을 만들 수도 있다.

``` r
gam.lo.i = gam(wage~lo(year,age,span=0.5)+education,data=Wage)
```

    ## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
    ## bf.maxit, : liv too small. (Discovered by lowesd)

    ## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
    ## bf.maxit, : lv too small. (Discovered by lowesd)

    ## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
    ## bf.maxit, : liv too small. (Discovered by lowesd)

    ## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
    ## bf.maxit, : lv too small. (Discovered by lowesd)

``` r
library(akima) # akima library을 사용하여 plot을 그린다.  
plot(gam.lo.i)
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-24-1.png)![](7장_Lab_files/figure-markdown_github/unnamed-chunk-24-2.png)

GAM에서 logistic을 사용하려면 I()을 binary response variable을 만들기 위해 사용하면 된다.

``` r
gam.lr = gam(I(wage>250)~year+s(age,df=5)+education,data=Wage,family=binomial)
par(mfrow=c(1,3))
plot(gam.lr,se=TRUE,col="green")
```

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-25-1.png)

<HS 카테고리에 high earners가 없음을 확인할 수 있다.


```r
table(education,I(wage>250))

education FALSE TRUE
--------------------

1. &lt; HS Grad 268 0
---------------------

2. HS Grad 966 5
----------------

3. Some College 643 7
---------------------

4. College Grad 663 22
----------------------

5. Advanced Degree 381 45
-------------------------

\`\`\`

따라서 <HS 범주를 제외하고 다시 logistic reg을 해본다.


```r
gam.lr.s = gam(I(wage>250)~year+s(age,df=5)+education,data=Wage,subset=(education != "1. &lt; HS Grad"),family=binomial) par(mfrow=c(1,3)) plot(gam.lr.s,se=TRUE,col="green") \`\`\`

![](7장_Lab_files/figure-markdown_github/unnamed-chunk-27-1.png)
