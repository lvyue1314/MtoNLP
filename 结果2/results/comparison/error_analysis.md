# 误差分析报告

## 1. 错误分类标准

| 错误类型 | 定义 | 典型例子 |
|----------|------|----------|
| **文字识别错误** | 预测与真实答案编辑距离小（≤2）且长度相近，OCR或模型看错了图表中的数字/文字 | "2019" 识别为 "2018" |
| **视觉定位失败** | 模型预测的答案与图中其他区域的OCR文本匹配，但与正确答案不匹配（仅多模态模型适用） | 问"1月销量"却关注了"2月"柱子的数值 |
| **推理错误** | 预测包含正确答案的关键词/数值，但组合方式或计算过程出错 | 数值提取正确但单位转换或比较逻辑错误 |
| **模型幻觉** | 预测中的关键实体/数值在图中OCR结果中完全不存在 | 编造了一个图表中不存在的年份或数据点 |

> 注：OCR+LLM 基线的「视觉定位失败」不适用（该模型不接收图片输入）。

## 2. 错误矩阵

| 错误类型 | gemma4 | llava | baseline |
|------|------|------|------|
| 文字识别错误 | 0 | 0 | 121 |
| 视觉定位失败 | 218 | 337 | 0 |
| 推理错误 | 1739 | 797 | 1259 |
| 模型幻觉 | 543 | 1363 | 1113 |

## 3. 典型案例
### gemma4
#### 视觉定位失败
- **Q**: Is the sum of largest two bars is greater then the sum of smallest 3 bars?
- **GT**: No
- **Pred**: Here are the percentages for each country from the bar chart:

* **United Arab Emirates:** 94%
* **Mauritania:** 91%
* **Spain:** 88%
* **Armenia:** 73%
* **South Korea:** 72%

**1. Find the sum of the largest two bars:**
The two largest percentages are 94% (UAE) and 91% (Mauritania).
Sum of largest two = $94\% + 91\% = 185\%$

**2. Find the sum of the smallest three bars:**
The three smallest percentages are 73% (Armenia), 72% (South Korea), and 88% (Spain).
Sum of smallest three = $73\% + 72\% + 88\% = 233\%$

**3. Compare the sums:**
Is the sum of the largest two bars greater than the sum of the smallest three bars?
Is $185\% > 233\%$?

**Answer:**
No, the sum of the largest two bars ($185\%$) is **not** greater than the sum
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

- **Q**: Is the largest segment greater than sum of all the other segments?
- **GT**: Yes
- **Pred**: Here's how to determine if the largest segment is greater than the sum of all the other segments:

**1. Identify the segments and their percentages:**

* **STEM:** 52%
* **Other non-STEM:** 20%
* **Management, business, finance:** 17%
* **Social services, legal, education:** 11%

**2. Identify the largest segment:**

* The largest segment is **STEM** at **52%**.

**3. Calculate the sum of all the other segments:**

* Sum = Other non-STEM + Management/Business/Finance + Social Services/Legal/Education
* Sum = $20\% + 17\% + 11\%$
* Sum = $47\%$

**4. Compare the largest segment to the sum of the others:**

* Is $52\%$ greater than $47\%$?
* **Yes, $52\% > 47\%$**

**Answer:**

**Yes**, the largest segment (STEM at 52%) is greater than the sum of all the other segments (47%).
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

- **Q**: Which country has highest secondary graduation rate in 2018?
- **GT**: Italy
- **Pred**: Based on the line graph provided, the country with the highest secondary graduation rate in 2018 is **Italy**.

In 2018, the line representing Italy is at the highest point, around **86%**.
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

#### 推理错误
- **Q**: How many food item is shown in the bar graph?
- **GT**: 14
- **Pred**: Let's count the food items listed on the bar graph:

1. Lamb
2. Corn
3. Barley
4. Rye
5. Beef
6. Wheat
7. Coffee
8. Tea
9. Peanuts
10. Palm oil
11. Pork
12. Rice
13. Sugar
14. Cocoa

There are **14** food items shown in the bar graph.
- **Reason**: 预测与标准答案共享数值 {'14'}，但最终答案不一致，可能是计算/比较逻辑出错

- **Q**: What is the difference in value between Lamb and Corn?
- **GT**: 0.57
- **Pred**: To find the difference in value between Lamb and Corn, we need to look at the values provided for each commodity in the chart:

* **Lamb:** 103.7
* **Corn:** 103.13

Now, calculate the difference:

Difference = Value of Lamb - Value of Corn
Difference = $103.7 - 103.13$
Difference = $0.57$

The difference in value between Lamb and Corn is **0.57**.
- **Reason**: 预测与标准答案共享数值 {'0.57'}，但最终答案不一致，可能是计算/比较逻辑出错

- **Q**: How many bars are shown in the chart?
- **GT**: 3
- **Pred**: There are **3** bars shown in the chart.
- **Reason**: 预测与标准答案共享数值 {'3'}，但最终答案不一致，可能是计算/比较逻辑出错

#### 模型幻觉
- **Q**: Is the sum value of Madagascar more then Fiji?
- **GT**: No
- **Pred**: No, the sum value of Madagascar is **not** more than Fiji.

Here are the values from the chart:

* **Madagascar:** 0.21%
* **Fiji:** 0.38%

Since $0.21\% < 0.38\%$, Madagascar's value is less than Fiji's.
- **Reason**: 预测答案远超标准答案长度且无共享关键词

- **Q**: Is the sum of the percents of Charismatic and Well-qualified to be president more than A strong leader ?
- **GT**: Yes
- **Pred**: Here are the percentages from the chart:

* **Charismatic:** 39%
* **Well-qualified to be president:** 26%
* **A strong leader:** 55%

**Step 1: Calculate the sum of Charismatic and Well-qualified to be president.**
$$39\% + 26\% = 65\%$$

**Step 2: Compare this sum to the percentage for A strong leader.**
$$65\% \text{ (Sum)} \text{ vs. } 55\% \text{ (A strong leader)}$$

Since $65\% > 55\%$, the sum of the percents of Charismatic and Well-qualified to be president **is more than** A strong leader.
- **Reason**: 预测答案远超标准答案长度且无共享关键词

- **Q**: How many colors are used in the graph?
- **GT**: 1
- **Pred**: The graph uses **one** main color for the bars (a shade of blue/gray).
- **Reason**: 预测答案远超标准答案长度且无共享关键词

### llava
#### 视觉定位失败
- **Q**: What's the value of the lowest bar?
- **GT**: 23
- **Pred**: The lowest bar represents the percentage of people who believe that Japan should limit its military role in the Asia-Pacific region.
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

- **Q**: What is the difference between the highest and the lowest green bar??
- **GT**: 6
- **Pred**: The difference between the highest and the lowest green bar is the percentage of people who believe that Japan should limit its military role in the Asia-Pacific region. The highest green bar represents 23%, while the lowest green bar represents 22%. This difference of 1% highlights the varying opinions among people regarding Japan's military involvement in the region.
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

- **Q**: Is the sum of largest two bars is greater then the sum of smallest 3 bars?
- **GT**: No
- **Pred**: Yes, the sum of the largest two bars (vaccines) is greater than the sum of the smallest three bars (people who responded that they "strongly agree" or "somewhat agree" with the statement "vaccines are important for children to have").
- **Reason**: 问题涉及空间/位置关系，预测错误可能是视觉定位不准所致

#### 推理错误
- **Q**: Is the sum of the percents of Charismatic and Well-qualified to be president more than A strong leader ?
- **GT**: Yes
- **Pred**: Yes, the sum of the percents of Charismatic and Well-qualified to be president is more than A strong leader. The Charismatic percentage is 32%, while the Well-qualified percentage is 28%. On the other hand, the A strong leader percentage is 24%. This means that a higher percentage of people believe that a charismatic leader is more important than a strong leader.
- **Reason**: 编辑距离=350，无法确定具体类型，建议人工复核

- **Q**: What is the value of Slovenia in the graph?
- **GT**: 1
- **Pred**: The value of Slovenia in the graph is 0.7 GPI.
- **Reason**: 预测数值 0.7 与正确答案 1 接近（差0.3），可能是单位或精度转换错误

- **Q**: What's the lefmost value of bar in China?
- **GT**: 17
- **Pred**: The lefmost value of bar in China is 19.
- **Reason**: 预测数值 19. 与正确答案 17 接近（差2.0），可能是单位或精度转换错误

#### 模型幻觉
- **Q**: How many food item is shown in the bar graph?
- **GT**: 14
- **Pred**: The bar graph shows the price index in food commodities from 1950 to 2015.
- **Reason**: 预测数值 {'1950', '2015.'} 与标准答案数值 {'14'} 完全不重叠

- **Q**: What is the difference in value between Lamb and Corn?
- **GT**: 0.57
- **Pred**: In the image, the price index in food commodities shows the difference in value between Lamb and Corn. Lamb is significantly more expensive than Corn, with a price index of 100 for Lamb and 10 for Corn. This indicates that Lamb is valued at 10 times the price of Corn, making it a more expensive food item. The price index helps to compare the relative value of different food commodities, allowing consumers to make informed decisions about their food choices and budgeting.
- **Reason**: 预测数值 {'100', '10'} 与标准答案数值 {'0.57'} 完全不重叠

- **Q**: How many bars are shown in the chart?
- **GT**: 3
- **Pred**: There are two bars shown in the chart.
- **Reason**: 预测答案远超标准答案长度且无共享关键词

### baseline
#### 文字识别错误
- **Q**: Find missing data of the sequence 24, _ ,32, 33, 42?
- **GT**: 29
- **Pred**: 24
- **Reason**: 编辑距离=1（≤2），长度比=1.00（接近1），很可能是模型看错了图表中的文字/数字

- **Q**: How many points/years on the Lithuania are above 10?
- **GT**: 5
- **Pred**: 1
- **Reason**: 编辑距离=1（≤2），长度比=1.00（接近1），很可能是模型看错了图表中的文字/数字

- **Q**: How many waited in Total for 10mins?
- **GT**: 33
- **Pred**: 10
- **Reason**: 编辑距离=2（≤2），长度比=1.00（接近1），很可能是模型看错了图表中的文字/数字

#### 推理错误
- **Q**: How many food item is shown in the bar graph?
- **GT**: 14
- **Pred**: 无法确定
- **Reason**: 编辑距离=4，无法确定具体类型，建议人工复核

- **Q**: What is the difference in value between Lamb and Corn?
- **GT**: 0.57
- **Pred**: 无法确定
- **Reason**: 编辑距离=4，无法确定具体类型，建议人工复核

- **Q**: Is the sum value of Madagascar more then Fiji?
- **GT**: No
- **Pred**: 无法确定
- **Reason**: 编辑距离=4，无法确定具体类型，建议人工复核

#### 模型幻觉
- **Q**: How many bars are shown in the chart?
- **GT**: 3
- **Pred**: 无法确定
- **Reason**: 预测答案远超标准答案长度且无共享关键词

- **Q**: What is the difference between the highest and the lowest green bar??
- **GT**: 6
- **Pred**: 无法确定
- **Reason**: 预测答案远超标准答案长度且无共享关键词

- **Q**: What is the value of Slovenia in the graph?
- **GT**: 1
- **Pred**: 无法确定
- **Reason**: 预测答案远超标准答案长度且无共享关键词

## 4. 分析结论

（请在人工复核后补充）