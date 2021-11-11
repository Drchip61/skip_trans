# python集合与列表查找差异

### 1. 集合与列表查找对比
* 关于大量数据查找，效率差距到底有多大？

  先看一组实例：
  ```bash
  import time
  import numpy as np

  nums = np.random.randint( 0, 1e7, int(1e3))
  set1 = set(np.random.randint(0, 1e7, int(1e5)))
  list1 = list(set1)

  t1 = time.time()
  for i in nums:
      i in set1
  t2 = time.time()
  for i in nums:
      i in list1
  t3 = time.time()
  print(t2-t1)
  print(t3-t2)
  ```
  
  结果展示：
  ```bash
  0.0009751319885253906
  3.9837825298309326
  ```

* 查找效率差距巨大
  对于大数据集量来说，我们清晰的看到，集合的查找效率远远的高于列表，那么本文接下来会从python底层数据结构的角度分析为何出现如此情况。



### 2. Python List数据结构

Python中的list作为一个常用数据结构，在很多程序中被用来当做数组使用，可能很多人都觉得list无非就是一个动态数组，就像C++中的vector或者Go中的slice一样。但事实真的是这样的吗？

我们来思考一个简单的问题，Python中的list允许我们存储不同类型的数据，既然类型不同，那**内存占用空间就就不同，不同大小的数据对象**又是如何"存入"数组中呢？

比如下面的代码中，我们分别在数组中存储了一个字符串，一个整形，以及一个字典对象，假如是数组实现，则需要将数据存储在相邻的内存空间中，而索引访问就变成一个相当困难的事情了，毕竟我们无法猜测每个元素的大小，从而无法定位想要的元素位置。

```bash

>>> a = "hello world"; b = 456; c = {}
>>> d = [a, b, c]
 
>>> d
['hello world', 456, {}]
```

是否是通过链表结构实现的呢? 毕竟链表支持动态的调整，借助于指针可以引用不同类型的数据，比如下面的图示中的链表结构。但是这样的话使用下标索引数据的时候，需要依赖于遍历的方式查找，O(n)的时间复杂度访问效率实在是太低。

同时使用链表的开销也较大，每个数据项除了维护本地数据指针外，还要维护一个next指针，因此还要额外分配8字节数据，同时链表分散性使其无法像数组一样利用CPU的缓存来高效的执行数据读写。

* 实现的细节可以从其Python的源码中找到， 定义如下：
```bash

typedef struct {
  PyObject_VAR_HEAD
  PyObject **ob_item;
  Py_ssize_t allocated;
} PyListObject;
```

内部list的实现的是一个C结构体，该结构体中的ob_item是一个指针数组，存储了所有对象的指针数据，allocated是已分配内存的数量, PyObject_VAR_HEAD是一个宏扩展包含了更多扩展属性用于管理数组，比如引用计数以及数组大小等内容。

**所以我们可以看出，用动态数组作为第一层数据结构，动态数组里存储的是指针，指向对应的数据**

* 动态数组
既然是一个动态数组，则必然会面临一个问题，如何进行容量的管理，大部分的程序语言对于此类结构使用动态调整策略，也就是当存储容量达到一定阈值的时候，扩展容量，当存储容量低于一定的阈值的时候，缩减容量。

道理很简单，但实施起来可没那么容易，什么时候扩容，扩多少，什么时候执行回收，每次又要回收多少空闲容量，这些都是在实现过程中需要明确的问题。

假如我们使用一种最简单的策略：超出容量加倍，低于一半容量减倍。这种策略会有什么问题呢？设想一下当我们在容量已满的时候进行一次插入，随即删除该元素，交替执行多次，那数组数据岂不是会不断的被整体复制和回收，已经无性能可言了。

对于Python list的动态调整规则程序中定义如下, 当追加数据容量已满的时候，通过下面的方式计算再次分配的空间大小，创建新的数组，并将所有数据复制到新的数组中。这是一种相对数据增速较慢的策略，回收的时候则当容量空闲一半的时候执行策略，获取新的缩减后容量大小。

具体规则如下：
```bash

new_allocated = (newsize >> 3) + (newsize < 9 ? 3 : 6);
 
new_allocated += newsize
```
动态数组扩容规则是：当出现数组存满时，扩充容量新加入的长度和额外3个，如果新加入元素大于9时，则扩6额外。


其实对于Python列表这种数据结构的动态调整，在其他语言中也都存在，只是大家可能在日常使用中并没有意识到，了解了动态调整规则，我们可以通过比如手动分配足够的空间，来减少其动态分配带来的迁移成本，使得程序运行的更高效。

另外如果事先知道存储在列表中的数据类型都相同，比如都是整形或者字符等类型，可以考虑使用arrays库，或者numpy库，两者都提供更直接的数组内存存储模型，而不是上面的指针引用模型，因此在访问和存储效率上面会更高效一些。



### 3. raise用法
* raise语法格式

```bash
raise [exceptionName[(reason)]]
```

等价于

```bash
raise  # 该语句引发当前上下文中捕获的异常（比如在 except 块中），或默认引发 RuntimeError 异常。
raise exceptionName  # 表示引发执行类型的异常。
raise exceptionName(reason)   # 在引发指定类型的异常的同时，附带异常的描述信息。
```

举例1：无参数raise
```bash
>>> raise
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    raise
RuntimeError: No active exception to reraise
```
举例2：raise exceptionName
```bash
>>> raise ZeroDivisionError
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    raise ZeroDivisionError
ZeroDivisionError
```
举例3：raise exceptionName（reason）：
```bash
>>> raise ZeroDivisionError('除数不能为零')
Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    raise ZeroDivisionError('除数不能为零')
ZeroDivisionError: 除数不能为零
```

* 配合try主动引发异常
raise 语句引发的异常通常用 try except（else finally）异常处理结构来捕获并进行处理。
使用 raise 语句引发异常，程序的执行是正常的，手动抛出的异常并不会导致程序崩溃。
  * 示例：
  ```bash
  try:
    num1 = int(input('输入一个被除数 num1：'))  # 用户输入一个被除数
    num2 = int(input('输入一个除数 num2：'))  # 用户输入一个除数
    result = num1 / num2
    # 判断用户输入的除数是否为零
    if (num2 == 0):
        raise ZeroDivisionError
except ZeroDivisionError as e:
    print('引发异常：', repr(e))

```
* 运行结果：
```bash
输入一个被除数 num1：6
输入一个被除数 num2：0
引发异常： ZeroDivisionError('division by zero')
```
* 自定义异常
你可以通过创建一个新的异常类来拥有自己的异常。异常类继承自 Exception 类，可以直接继承，或者间接继承，例如：
```bash
class MyError(Exception):
   def __init__(self, value):
       self.value = value
       return repr(self.value)
   # raise MyError('oops!')
   print('My exception occurred, value:', e.value)
```

### 4. assert用法

设想一个情况，我们的代码中包含数据读取和数据处理的部分，其中数据处理需要GPU计算，而数据读取也需要大量时间，甚至读取数据之后还需要进一步的处理，如果我们在处理完之后才发现GPU不可用，就大大降低了效率，故而需要一种提前判断的方式去解决此问题。

因此我们提出断言assert！

Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。


assert语法：
```bash
assert expression
```
等价于：
```bash
if not expression:
    raise AssertionError(arguments)
```
实例：
```bash
import torch
assert (torch.cuda.is_available()), "本机器需要有可用的GPU才能运行此代码！"
```
### 5.Traceback用法

Trachback是用来获取异常的详细信息的。
try…except…的输出结果只能让你知道报了这个错误，却不知道在哪个文件哪个函数哪一行报的错。使用 traceback 模块可以非常清楚的了解具体错误内容在哪。
* Python程序的traceback信息均来源于一个叫做traceback object的对象，而这个traceback object通常是通过函数sys.exc_info()来获取的。
* sys.exc_info()获取了当前处理的exception的相关信息，并返回一个元组。
  * 元组的第一个数据是异常的类型.
  * 第二个返回值是异常的value值.
  * 第三个就是我们要的traceback object.
示例：
```bash
import sys

def func1():
    raise Exception("--func1 exception--")
    
def test():
    try:
        func1()
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        print("exc_type: %s" % exc_type)
        print("exc_value: %s" % exc_value)
        print("exc_traceback_obj: %s" % exc_traceback_obj)
 
 test()
```
结果：
```bash
exc_type: <class 'Exception'>
exc_value: --func1 exception--
exc_traceback_obj: <traceback object at 0x0000024D2F6A22C8>

Process finished with exit code 0
```

## Reference
* [怕蛇的人怎么学python](https://mp.weixin.qq.com/s?__biz=MzA4Nzg3Njg1OA==&mid=2247484069&idx=1&sn=ebb352dc62949fcbd79a397f2657b235&chksm=9033f730a7447e26bcdeb8b08834fa335c5a194f1ac0864e2ea9ff068c72fde09b32f11f2153&mpshare=1&scene=23&srcid=1019G5KQ4IZVJZC6kOJmHRhD&sharer_sharetime=1635991918869&sharer_shareid=4bbdc95dbeb4de0f49bd4127857cc1c2#rd)
* [python raise语句详解](https://blog.csdn.net/manongajie/article/details/106288078?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163600172016780265497497%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163600172016780265497497&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-106288078.pc_search_result_hbase_insert&utm_term=raise%E6%89%8B%E5%8A%A8%E5%BC%95%E5%8F%91%E5%BC%82%E5%B8%B8&spm=1018.2226.3001.4187)
* [Traceback异常打印](https://blog.csdn.net/aiao34980/article/details/101488938?ops_request_misc=&request_id=&biz_id=102&utm_term=traceback&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-101488938.pc_search_result_hbase_insert&spm=1018.2226.3001.4187)

## Citations

```bibtex
@
  title={Python异常捕获与处理},
  author={Tianyu Yan},
  date={2021/11/4}
}
```
