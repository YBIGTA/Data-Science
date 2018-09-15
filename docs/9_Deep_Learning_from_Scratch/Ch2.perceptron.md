{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chapter 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 퍼셉트론"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "을 시작하기에 앞서, 저희 팀의 스터디 계획을 간단하게 소개하겠습니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 퍼셉트론이란?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "왜 퍼셉트론인가? : 신경망의 기원이 되는 알고리즘이기 때문이다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "퍼셉트론 : 다수의 신호를 입력으로 받아 하나의 신호를 출력하는 논리구조 (이 때의 신호는 0과 1 두 가지의 값만 가질 수 있다.)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/sW1qY7B.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "위의 그림은 입력으로 두 개의 신호를 받은 퍼셉트론의 예시이다. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "x1, x2 : 입력 신호\n",
    "\n",
    "y : 출력 신호\n",
    "\n",
    "w1, w2 : 가중치"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "원은 뉴런(혹은 노드)라고 부르는데, 입력 신호가 뉴런에 보내질 때는 각각 고유한 **가중치**가 곱해진다(w1x1, w2x2). 이 신호의 총합이 정해진 한계를 넘어설 때, 다음 뉴런에서는 1을 출력한다(흔히 말하는 '뉴런이 활성화한다'고 표현). 그 한계값을 **임계값**이라고 하며, $\\theta$로 기호를 나타낸다. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "이를 수식으로 나타내면 다음과 같다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/7tXWRU7.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 단순한 논리 회로"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.2.1 AND 게이트"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/UNJJQRZ.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "AND 게이트를 어떻게 퍼셉트론으로 표현할까? 위의 표대로 작동하도록, 우리는 매개변수인 가중치와 임계치의 값을 정해야 한다. 이를 나타내는 매개변수의 조합은 무수히 많다. 가령 (w1, w2, $\\theta$) = (0.5, 0.5, 0.7)일 때도 가능하고, (1.0, 1.0, 1.0)일 때도 가능하다. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.2.2 NAND 게이트와 OR 게이트"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/aRNuD3V.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NAND 게이트를 표현하는 매개변수의 조합은 (w1, w2, $\\theta$) = (-0.5, -0.5, -0.7)인 경우가 있다. 사실 AND 게이트의 매개변수의 부호를 모두 반전하면 NAND 게이트가 된다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/Mgs1VeG.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "OR 게이트를 표현하는 매개변수의 조합은 다양하지만, (w1, w2, $\\theta$) = (1.2, 1.2, 1)이 가능할 것이다. 퍼셉트론의 구조는 AND, NAND, OR 게이트 모두 동일하며, 매개변수(가중치 및 임계값)의 값만 다르다. 이를 확장시키면, 복잡한 논리연산도 표현 할 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.3 퍼셉트론 구현하기"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.3.1 간단한 구현부터"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "AND 게이트를 함수로 구현하면 다음과 같다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "def AND(x1, x2):\n",
    "    w1, w2, theta = 0.5, 0.5, 0.7\n",
    "    tmp = x1*w1 + x2*w2\n",
    "    if tmp <= theta:\n",
    "        return 0\n",
    "    else:\n",
    "        return 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(0, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(0, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(1, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.3.2 가중치와 편향 도입"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "위에서 표현한 퍼셉트론의 식은 아래와 같다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/7tXWRU7.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "여기에서 $\\theta$를 좌변으로 넘겨서 -b로 치환하면 식은 아래와 같이 변한다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/Wnqcf3h.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "여기에서 b를 **편향**이라고 부른다. 퍼셉트론은 입력신호에 가중치를 곱한 값과(w1x1, w2x2)와 편향(b)를 합하여, 그 값이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력한다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "가중치는 각 입력 신호가 결과에 주는 영향력(중요도)를 조절하는 매개변수이고, 편향은 뉴런이 얼마나 쉽게 활성화(1을 출력)하는지 조정하는 매개변수이다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.3.3 가중치와 편향 구하기"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "AND 게이트를 다시 구현하면 다음과 같다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def AND(x1, x2):\n",
    "    x = np.array([x1, x2])\n",
    "    w = np.array([0.5, 0.5])\n",
    "    b = -0.7\n",
    "    tmp = np.sum(x*w) + b\n",
    "    if tmp <= 0:\n",
    "        return 0\n",
    "    else:\n",
    "        return 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(0,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(0,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "AND(1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "NAND 게이트는 AND 게이트의 매개변수에 -1을 곱해주면 된다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "def NAND(x1, x2):\n",
    "    x = np.array([x1, x2])\n",
    "    w = np.array([-0.5, -0.5])\n",
    "    b = 0.7\n",
    "    tmp = np.sum(x*w) + b\n",
    "    if tmp <= 0:\n",
    "        return 0\n",
    "    else:\n",
    "        return 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NAND(0,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NAND(1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NAND(1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NAND(1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "OR 게이트를 구현해보자."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def OR(x1, x2):\n",
    "    x = np.array([x1, x2])\n",
    "    w = np.array([0.5, 0.5])\n",
    "    b = -0.2\n",
    "    tmp = np.sum(x*w) + b\n",
    "    if tmp <= 0:\n",
    "        return 0\n",
    "    else:\n",
    "        return 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "OR(0,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "OR(1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "OR(0,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "OR(1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "각 함수를 보면 AND와 NAND, OR 함수는 모두 같은 구조로 되어있고, 차이는 가중치와 편향 매개변수의 값 뿐임을 알 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.4 퍼셉트론의 한계"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.4.1 XOR 게이트"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/xqo8Sgr.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "XOR 게이트는 지금까지의 퍼셉트론으로는 구현할 수 없다. 왜일까? 아래의 그래프를 보면 이해하기 쉽다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/GkZqI63.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "우리가 지금까지 본 퍼셉트론은 위의 그래프에 하나의 직선으로 0과 1을 구분할 수 있다. 하지만 XOR 게이트는 하나의 직선으로 두 영역을 구분할 수 없다. 그래서 하나의 퍼셉트론으는 구현할 수 없다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.4.2 선형과 비선형"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/P323uHu.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "위의 그래프처럼, 곡선이라면 둘을 나눌 수 있다. 퍼셉트론은 직선 하나로 나눈 영역만 표현할 수 있다는 한계가 있다. 그렇다면 퍼셉트론을 여러 개 사용하면, XOR 게이트도 구현할 수 있지 않을까?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.5 다층 퍼셉트론"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "퍼셉트론의 장점은 '층을 쌓아' **다층 퍼셉트론**을 만들 수 있다는 것이다. 층을 여러 개 쌓으면, 보다 더 복잡한 문제도 퍼셉트론으로 구현할 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.5.1 기존 게이트 조합하기"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "기존에 만든 AND, NAND, OR 게이트를 조합하면 XOR 게이트를 만들 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/76p3Mbe.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "위의 그래프를 표 형식으로 나타내보자."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![Imgur](https://i.imgur.com/tgJA7E9.jpg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2.5.2 XOR 게이트 구현하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "def XOR(x1, x2):\n",
    "    s1 = NAND(x1, x2)\n",
    "    s2 = OR(x1, x2)\n",
    "    y = AND(s1, s2)\n",
    "    return y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XOR(0,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XOR(1,0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XOR(0,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XOR(1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "이렇게 퍼셉트론은 층을 쌓으면 보다 더 복잡하고 다양한 것을 표현할 수 있다."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
