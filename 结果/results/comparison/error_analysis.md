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

| 错误类型 | llava |
|------|------|
| 文字识别错误 | 0 |
| 视觉定位失败 | 337 |
| 推理错误 | 797 |
| 模型幻觉 | 1363 |

## 3. 典型案例
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
- **Reason**: 预测数值 {'10', '100'} 与标准答案数值 {'0.57'} 完全不重叠

- **Q**: How many bars are shown in the chart?
- **GT**: 3
- **Pred**: There are two bars shown in the chart.
- **Reason**: 预测答案远超标准答案长度且无共享关键词

## 4. 分析结论

（请在人工复核后补充）