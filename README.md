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

既然是一个动态数组，则必然会面临一个问题，如何进行容量的管理，大部分的程序语言对于此类结构使用动态调整策略，**也就是当存储容量达到一定阈值的时候，扩展容量，当存储容量低于一定的阈值的时候，缩减容量**。

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

* 查找原理

从上面的数据结构可以得出，python list的查找时间复杂度为O（n），因为作为一个动态数组，需要遍历每一个元素去找到目标元素，故而是一种较为低效的查找方式。

### 3. Python set数据结构

说到集合，就不得不提到Python中的另一种数据结构，就是字典。**字典和集合有异曲同工之妙**。

在Python中，字典是通过散列表或说哈希表实现的。字典也被称为关联数组，还称为哈希数组等。也就是说，字典也是一个数组，但数组的索引是键经过哈希函数处理后得到的散列值。哈希函数的目的是使键均匀地分布在数组中，并且可以在内存中以O(1)的时间复杂度进行寻址，从而实现快速查找和修改。哈希表中哈希函数的设计困难在于将数据均匀分布在哈希表中，从而尽量减少哈希碰撞和冲突。由于不同的键可能具有相同的哈希值，即可能出现冲突，高级的哈希函数能够使冲突数目最小化。Python中并不包含这样高级的哈希函数，几个重要（用于处理字符串和整数）的哈希函数是常见的几个类型。通常情况下建立哈希表的具体过程如下：

数据添加：把key通过哈希函数转换成一个整型数字，然后就将该数字对数组长度进行取余，取余结果就当作数组的下标，将value存储在以该数字为下标的数组空间里。
数据查询：再次使用哈希函数将key转换为对应的数组下标，并定位到数组的位置获取value。
哈希函数就是一个映射，因此哈希函数的设定很灵活，只要使得任何关键字由此所得的哈希函数值都落在表长允许的范围之内即可。本质上看哈希函数不可能做成一个一对一的映射关系，其本质是一个多对一的映射，这也就引出了下面一个概念–哈希冲突或者说哈希碰撞。哈希碰撞是不可避免的，但是一个好的哈希函数的设计需要尽量避免哈希碰撞。

Python中使用使用开放地址法解决冲突。

CPython使用伪随机探测(pseudo-random probing)的散列表(hash table)作为字典的底层数据结构。由于这个实现细节，只有可哈希的对象才能作为字典的键。字典的三个基本操作（添加元素，获取元素和删除元素）的平均事件复杂度为O(1)。

```bash
Python中所有不可变的内置类型都是可哈希的。
可变类型（如列表，字典和集合）就是不可哈希的，因此不能作为字典的键。
```

常见的哈希碰撞解决方法：

* 1 开放寻址法（open addressing）

  开放寻址法中，所有的元素都存放在散列表里，当产生哈希冲突时，通过一个探测函数计算出下一个候选位置，如果下一个获选位置还是有冲突，那么不断通过探测函数往下找，直到找个一个空槽来存放待插入元素。

  开放地址的意思是除了哈希函数得出的地址可用，当出现冲突的时候其他的地址也一样可用，常见的开放地址思想的方法有线性探测再散列，二次探测再散列等，这些方法都是在第一选择被占用的情况下的解决方法。

* 2 再哈希法

  这个方法是按顺序规定多个哈希函数，每次查询的时候按顺序调用哈希函数，调用到第一个为空的时候返回不存在，调用到此键的时候返回其值。

* 3 链地址法

  将所有关键字哈希值相同的记录都存在同一线性链表中，这样不需要占用其他的哈希地址，相同的哈希值在一条链表上，按顺序遍历就可以找到。

* 4 公共溢出区

   其基本思想是：所有关键字和基本表中关键字为相同哈希值的记录，不管他们由哈希函数得到的哈希地址是什么，一旦发生冲突，都填入溢出表。

* 5 装填因子α

  一般情况下，处理冲突方法相同的哈希表，其平均查找长度依赖于哈希表的装填因子。哈希表的装填因子定义为表中填入的记录数和哈希表长度的比值，也就是标志着哈希表的装满程度。直观看来，α越小，发生冲突的可能性就越小，反之越大。一般0.75比较合适，涉及数学推导。

在python中一个key-value是一个entry，

entry有三种状态。
```bash
Unused： me_key == me_value == NULL

Unused是entry的初始状态，key和value都为NULL。插入元素时，Unused状态转换成Active状态。这是me_key为NULL的唯一情况。

Active： me_key != NULL and me_key != dummy 且 me_value != NULL

插入元素后，entry就成了Active状态，这是me_value唯一不为NULL的情况，删除元素时Active状态刻转换成Dummy状态。

Dummy： me_key == dummy 且 me_value == NULL
```
此处的dummy对象实际上一个PyStringObject对象，仅作为指示标志。Dummy状态的元素可以在插入元素的时候将它变成Active状态，但它不可能再变成Unused状态。

为什么entry有Dummy状态呢？这是因为采用开放寻址法中，遇到哈希冲突时会找到下一个合适的位置，例如某元素经过哈希计算应该插入到A处，但是此时A处有元素的，通过探测函数计算得到下一个位置B，仍然有元素，直到找到位置C为止，此时ABC构成了探测链，查找元素时如果hash值相同，那么也是顺着这条探测链不断往后找，当删除探测链中的某个元素时，比如B，如果直接把B从哈希表中移除，即变成Unused状态，那么C就不可能再找到了，因为AC之间出现了断裂的现象，正是如此才出现了第三种状态---Dummy，Dummy是一种类似的伪删除方式，保证探测链的连续性。

set集合和dict一样也是基于散列表的，只是他的表元只包含键的引用，而没有对值的引用，其他的和dict基本上是一致的，所以在此就不再多说了。并且dict要求键必须是能被哈希的不可变对象，因此普通的set无法作为dict的键，必须选择被“冻结”的不可变集合类：frozenset。顾名思义，一旦初始化，集合内数据不可修改。

一般情况下普通的顺序表数组存储结构也可以认为是简单的哈希表，虽然没有采用哈希函数（取余），但同样可以在O(1)时间内进行查找和修改。但是这种方法存在两个问题：扩展性不强；浪费空间。

dict是用来存储键值对结构的数据的，set其实也是存储的键值对，只是默认键和值是相同的。Python中的dict和set都是通过散列表来实现的。下面来看与dict相关的几个比较重要的问题：

dict中的数据是无序存放的

操作的时间复杂度，插入、查找和删除都可以在O(1)的时间复杂度

这是因为查找相当于将查找值通过哈希函数运算之后，直接得到对应的桶位置（不考虑哈希冲突的理想情况下），故而复杂度为O（1）

由于键的限制，只有可哈希的对象才能作为字典的键和set的值。可hash的对象即python中的不可变对象和自定义的对象。可变对象(列表、字典、集合)是不能作为字典的键和st的值的。
与list相比：list的查找和删除的时间复杂度是O(n)，添加的时间复杂度是O(1)。但是dict使用hashtable内存的开销更大。为了保证较少的冲突，hashtable的装载因子，一般要小于0.75，在python中当装载因子达到2/3的时候就会自动进行扩容。



```

## Reference
* [python set和dict底层联系](https://blog.csdn.net/liuweiyuxiang/article/details/98943272?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163662590416780271512687%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163662590416780271512687&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-98943272.first_rank_v2_pc_rank_v29&utm_term=python%E9%9B%86%E5%90%88set%E5%BA%95%E5%B1%82%E5%8E%9F%E7%90%86&spm=1018.2226.3001.4187)
* [python List图截](https://blog.csdn.net/u014029783/article/details/107992840)


## Citations

```bibtex
@
  title={Python集合与列表查找原理},
  author={Tianyu Yan},
  date={2021/11/4}
}
```
