# GP 输出全为 0 的问题诊断

## 问题现象

从调试输出看到：
```
transform() 的结果:
  - factors_pred_train 形状: (5600, 50)
  - factors_pred_train 统计:
    * 均值: 0.000000
    * 标准差: 0.000000
    * 最小值: -0.000000
    * 最大值: 0.000000
    * 是否全为0: True
```

所有生成的因子值都是 0！

---

## 可能的原因

### 1. **输入特征 X_train 本身全为 0**

#### 检查方法
```python
# 在 main_gp_new.py 第 436-442 行已经有检查
print(f"X_train 统计信息:")
print(f"   - 均值: {np.mean(self.X_train):.6f}")
print(f"   - 标准差: {np.std(self.X_train):.6f}")
print(f"   - 是否全为0: {np.all(self.X_train == 0)}")
```

**如果 X_train 全为 0**：
- 问题出在数据加载或特征工程阶段
- 检查 `dataload.py` 中的特征生成逻辑
- 检查特征是否被错误地 norm 成了 0

#### 解决方案
```python
# 查看原始数据
print(f"\nX_train 前 5 行, 前 5 列:")
print(self.X_train[:5, :5])

# 查看每个特征的统计
for i in range(min(10, self.X_train.shape[1])):
    col = self.X_train[:, i]
    print(f"特征 {i}: min={col.min():.6f}, max={col.max():.6f}, std={col.std():.6f}")
```

---

### 2. **GP 生成的程序表达式有问题**

#### 检查方法
从调试代码（第 446-465 行）可以看到程序的表达式，需要查看：

```python
for i, prog in enumerate(self.est_gp._best_programs[:3]):
    print(f"   - 表达式: {str(prog)}")
    result = prog.execute(self.X_train)
    print(f"   - 执行结果统计: ...")
```

**可能的问题表达式：**

#### 问题 A：表达式是常数 0
```python
# 例如
"0.0"
"sub(X0, X0)"  # X0 - X0 = 0
"mul(X0, 0.0)" # X0 × 0 = 0
```

#### 问题 B：表达式中的 norm() 函数返回全 0
```python
# 如果输入全是相同值
X = [1, 1, 1, 1, 1]
norm(X) = (X - mean(X)) / std(X) = (1 - 1) / 0 = 0 / 0 = NaN → 替换为 0
```

#### 问题 C：表达式使用了错误的函数
```python
# 某些函数可能对特定输入返回 0
protected_division(0, X) = 0
protected_log(0.001 以下的值) = 0
```

---

### 3. **functions.py 中的 norm() 函数有 bug**

#### 检查 norm() 函数逻辑

```python
# functions.py 第 152-170 行
def norm(x, rolling_window=2000):
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    factors_std = factors_data.rolling(window=rolling_window, min_periods=1).std()
    factor_value = (factors_data) / factors_std
    
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    return np.nan_to_num(factor_value).flatten()
```

**潜在问题：**

#### 问题 A：标准差为 0
```python
# 如果数据前 rolling_window 个值都相同
X = [1, 1, 1, ..., 1]
std = 0
X / 0 = inf → 替换为 0
```

#### 问题 B：min_periods=1 导致前期标准差不准确
```python
# 第 1 个值：window size = 1
std([single_value]) = 0
# 导致前期都是 inf → 0
```

#### 问题 C：没有减去均值
```python
# 当前实现（第 164 行）
factor_value = (factors_data) / factors_std  # 只除以 std，没有减均值

# 如果数据全为正且接近均值
X = [0.5, 0.5, 0.5, ...]
std = 0
X / 0 = inf → 0
```

---

### 4. **transform() 方法中的执行逻辑有问题**

#### 检查 genetic.py 第 1736 行

```python
# genetic.py transform()
X_new = np.array([gp.execute(X) for gp in self._best_programs]).T
```

**可能的问题：**

#### 问题 A：_best_programs 为空
```python
if len(self._best_programs) == 0:
    X_new = np.array([]).T  # 空数组
```

#### 问题 B：execute() 返回 None
```python
# _program.py 第 395 行
# We should never get here
return None

# 如果程序结构有问题，可能返回 None
# np.array([None, None, ...]) → 转换可能出错
```

#### 问题 C：execute() 中的 terminals 提取错误
```python
# _program.py 第 384-386 行
terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
             else X[:, t] if isinstance(t, int)
             else t for t in apply_stack[-1][1:]]

# 如果 t 既不是 float 也不是 int，直接使用 t
# 可能导致错误的结果
```

---

## 诊断步骤

### Step 1: 检查输入数据

在 `main_gp_new.py` 的 `evaluate_factors()` 方法开头添加：

```python
def evaluate_factors(self):
    # 添加完整的输入数据检查
    print(f"\n{'='*80}")
    print(f"🔍 完整诊断：GP 输出为 0 的原因")
    print(f"{'='*80}\n")
    
    # 1. 检查输入数据
    print(f"1. 输入数据检查:")
    print(f"   X_train 形状: {self.X_train.shape}")
    print(f"   X_train 统计:")
    print(f"     - 均值: {np.mean(self.X_train):.6f}")
    print(f"     - 标准差: {np.std(self.X_train):.6f}")
    print(f"     - 最小值: {np.min(self.X_train):.6f}")
    print(f"     - 最大值: {np.max(self.X_train):.6f}")
    print(f"     - 零值占比: {np.sum(self.X_train == 0) / self.X_train.size * 100:.2f}%")
    print(f"     - NaN占比: {np.sum(np.isnan(self.X_train)) / self.X_train.size * 100:.2f}%")
    print(f"     - Inf占比: {np.sum(np.isinf(self.X_train)) / self.X_train.size * 100:.2f}%")
    
    # 检查每列特征
    print(f"\n   各特征统计（前 10 个）:")
    for i in range(min(10, self.X_train.shape[1])):
        col = self.X_train[:, i]
        print(f"   特征 {i} ({self.feature_names[i] if i < len(self.feature_names) else 'unknown'}):")
        print(f"     均值={np.mean(col):.6f}, std={np.std(col):.6f}, "
              f"min={np.min(col):.6f}, max={np.max(col):.6f}")
    
    print(f"\n   X_train 前 3 行, 前 5 列:")
    print(self.X_train[:3, :5])
```

### Step 2: 检查 GP 程序表达式

```python
    # 2. 检查 GP 生成的程序
    print(f"\n2. GP 程序检查:")
    print(f"   _best_programs 数量: {len(self.est_gp._best_programs)}")
    
    if len(self.est_gp._best_programs) > 0:
        for i, prog in enumerate(self.est_gp._best_programs[:5]):
            print(f"\n   程序 {i+1}:")
            print(f"   - 表达式: {str(prog)}")
            print(f"   - fitness: {prog.fitness_}")
            print(f"   - depth: {prog.depth_}")
            print(f"   - length: {prog.length_}")
            print(f"   - program 结构: {prog.program[:10]}...")  # 前 10 个节点
            
            # 手动执行
            try:
                result = prog.execute(self.X_train)
                print(f"   - 执行结果:")
                print(f"     类型: {type(result)}")
                print(f"     形状: {result.shape if hasattr(result, 'shape') else 'N/A'}")
                print(f"     均值: {np.mean(result) if result is not None else 'None':.6f}")
                print(f"     标准差: {np.std(result) if result is not None else 'None':.6f}")
                print(f"     最小值: {np.min(result) if result is not None else 'None':.6f}")
                print(f"     最大值: {np.max(result) if result is not None else 'None':.6f}")
                print(f"     是否全为0: {np.all(result == 0) if result is not None else 'N/A'}")
                print(f"     NaN数量: {np.sum(np.isnan(result)) if result is not None else 'N/A'}")
                print(f"     前 20 个值: {result[:20] if result is not None else 'None'}")
            except Exception as e:
                print(f"   - 执行出错: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
```

### Step 3: 测试 norm() 函数

```python
    # 3. 测试 norm() 函数
    print(f"\n3. 测试 norm() 函数:")
    from functions import norm
    
    # 测试用例 1：正常数据
    test_data_1 = np.random.randn(1000) * 10 + 5
    result_1 = norm(test_data_1, rolling_window=100)
    print(f"   测试 1 (正常随机数据):")
    print(f"     输入: 均值={np.mean(test_data_1):.6f}, std={np.std(test_data_1):.6f}")
    print(f"     输出: 均值={np.mean(result_1):.6f}, std={np.std(result_1):.6f}")
    print(f"     是否全为0: {np.all(result_1 == 0)}")
    
    # 测试用例 2：常数
    test_data_2 = np.ones(1000) * 5
    result_2 = norm(test_data_2, rolling_window=100)
    print(f"   测试 2 (常数数据):")
    print(f"     输入: 均值={np.mean(test_data_2):.6f}, std={np.std(test_data_2):.6f}")
    print(f"     输出: 均值={np.mean(result_2):.6f}, std={np.std(result_2):.6f}")
    print(f"     是否全为0: {np.all(result_2 == 0)}")
    
    # 测试用例 3：使用实际特征
    if self.X_train.shape[0] >= 100:
        test_data_3 = self.X_train[:, 0]  # 第一个特征
        result_3 = norm(test_data_3, rolling_window=100)
        print(f"   测试 3 (实际特征 0):")
        print(f"     输入: 均值={np.mean(test_data_3):.6f}, std={np.std(test_data_3):.6f}")
        print(f"     输出: 均值={np.mean(result_3):.6f}, std={np.std(result_3):.6f}")
        print(f"     是否全为0: {np.all(result_3 == 0)}")
```

### Step 4: 手动测试 transform()

```python
    # 4. 手动测试 transform()
    print(f"\n4. 测试 transform() 方法:")
    try:
        # 手动执行每个程序
        manual_results = []
        for i, prog in enumerate(self.est_gp._best_programs[:3]):
            result = prog.execute(self.X_train)
            manual_results.append(result)
            print(f"   程序 {i+1} 执行结果: 均值={np.mean(result):.6f}, 全为0={np.all(result==0)}")
        
        # 手动组合
        if len(manual_results) > 0:
            manual_transform = np.array(manual_results).T
            print(f"\n   手动 transform 结果:")
            print(f"     形状: {manual_transform.shape}")
            print(f"     均值: {np.mean(manual_transform):.6f}")
            print(f"     是否全为0: {np.all(manual_transform == 0)}")
        
        # 对比 est_gp.transform()
        official_transform = self.est_gp.transform(self.X_train)
        print(f"\n   官方 transform 结果:")
        print(f"     形状: {official_transform.shape}")
        print(f"     均值: {np.mean(official_transform):.6f}")
        print(f"     是否全为0: {np.all(official_transform == 0)}")
        
        # 对比是否一致
        if len(manual_results) > 0:
            is_same = np.allclose(manual_transform, official_transform)
            print(f"\n   手动 vs 官方: {'一致' if is_same else '不一致'}")
            
    except Exception as e:
        print(f"   测试出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}\n")
```

---

## 常见原因与解决方案

### 原因 1: norm() 函数的 rolling window 太大

**问题：**
```python
# 如果 rolling_window=2000 但数据只有 1000 个
# 前 2000 个数据的 std 计算不准确
```

**解决方案：**
```python
# 修改 functions.py norm() 函数
def norm(x, rolling_window=2000):
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    # 动态调整 window size
    actual_window = min(rolling_window, len(factors_data) // 4)
    if actual_window < 10:
        actual_window = min(10, len(factors_data))
    
    factors_std = factors_data.rolling(window=actual_window, min_periods=max(2, actual_window//10)).std()
    
    # 避免除以 0
    factors_std = factors_std.replace(0, 1e-8)
    
    factor_value = factors_data / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    
    return np.nan_to_num(factor_value).flatten()
```

### 原因 2: 输入特征本身方差太小

**问题：**
```python
# 特征值变化太小
X = [0.0001, 0.0001, 0.0001, ...]
std ≈ 0
norm(X) → 0
```

**解决方案：**
```python
# 检查特征生成逻辑
# 确保特征有足够的变化性

# 或者在 norm() 中添加保护
def norm(x, rolling_window=2000):
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    # 检查数据方差
    overall_std = factors_data.std().values[0]
    if overall_std < 1e-6:
        # 数据几乎无变化，直接返回 0
        logger.warning(f"norm(): 输入数据方差极小 (std={overall_std}), 返回全 0")
        return np.zeros(len(factors_data))
    
    # 正常处理
    ...
```

### 原因 3: GP 生成的程序有问题

**问题：**
```python
# GP 可能生成了这样的表达式
"norm(norm(norm(X0)))"  # 多重 norm 可能导致数值问题
"sub(X0, X0)"           # 自己减自己 = 0
"mul(X0, 0.0)"          # 乘以 0 = 0
```

**解决方案：**
```python
# 在 genetic.py 中添加表达式验证

def _validate_program(self, program):
    """验证程序是否会返回全 0"""
    # 用一小部分数据测试
    test_X = self.X[:100]
    result = program.execute(test_X)
    
    # 检查是否全为 0 或 NaN
    if np.all(result == 0) or np.all(np.isnan(result)):
        return False  # 拒绝这个程序
    
    # 检查是否方差太小
    if np.std(result) < 1e-6:
        return False
    
    return True
```

### 原因 4: fitness 函数返回了相同的 fitness

**问题：**
```python
# 如果所有程序的 fitness 都相同
# GP 可能选择了任意程序（包括返回 0 的程序）
```

**解决方案：**
```python
# 检查 fitness 的分布
print(f"Fitness 分布:")
fitnesses = [prog.fitness_ for prog in self.est_gp._best_programs]
print(f"  最小: {np.min(fitnesses)}")
print(f"  最大: {np.max(fitnesses)}")
print(f"  均值: {np.mean(fitnesses)}")
print(f"  标准差: {np.std(fitnesses)}")
```

---

## 快速修复建议

### 临时解决方案：使用 norm_log1p

```python
# 在 functions.py 中，将默认的 norm 函数临时改为 norm_log1p

# 找到所有使用 norm() 的地方
# 例如 functions.py 第 192 行
def _sigmoid(x1):
    with np.errstate(over='ignore', under='ignore'):
        # return norm(np.nan_to_num(1 / (1 + np.exp(-x1))))
        return norm_log1p(np.nan_to_num(1 / (1 + np.exp(-x1))))  # 临时修复
```

### 根本解决方案：改进 norm() 函数

```python
# 修改 functions.py 的 norm() 函数
def norm(x, rolling_window=2000):
    """
    改进的 norm 函数，更鲁棒
    """
    arr = np.asarray(x)
    
    # 1. 检查输入
    if len(arr) == 0:
        return np.array([])
    
    # 2. 清理异常值
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. 检查整体方差
    overall_std = np.std(arr)
    if overall_std < 1e-8:
        # 数据几乎无变化，返回全 0
        return np.zeros_like(arr)
    
    # 4. 动态调整窗口大小
    n = len(arr)
    actual_window = min(rolling_window, max(n // 4, 50))
    min_periods = max(10, actual_window // 10)
    
    # 5. 转为 DataFrame
    factors_data = pd.DataFrame(arr, columns=['factor'])
    
    # 6. 计算滚动统计
    factors_std = factors_data.rolling(
        window=actual_window, 
        min_periods=min_periods
    ).std()
    
    # 7. 避免除以 0
    factors_std = factors_std.replace(0, 1e-8)
    factors_std = factors_std.fillna(1.0)
    
    # 8. 标准化
    factor_value = factors_data / factors_std
    
    # 9. 再次清理
    factor_value = factor_value.replace([np.inf, -np.inf], 0.0)
    factor_value = factor_value.fillna(0.0)
    
    return factor_value.values.flatten()
```

---

## 执行诊断

在运行 GP 之前，在 `main_gp_new.py` 的 `evaluate_factors()` 方法中添加上述所有诊断代码，然后：

```bash
cd /Users/aming/project/python/crypto-workstation/gp_crypto_next
python main_gp_new.py  # 或你的运行命令
```

查看完整的诊断输出，定位到底是哪一步出了问题。

---

## 检查清单

- [ ] 输入数据 X_train 是否正常（不全为 0）
- [ ] GP 生成的表达式是否合理（不是常数 0）
- [ ] norm() 函数是否正确处理了数据
- [ ] execute() 方法是否正确执行了表达式
- [ ] transform() 方法是否正确组合了结果
- [ ] fitness 函数是否能区分不同程序
- [ ] rolling_window 大小是否合适

逐一排查这些问题，就能找到根本原因！

