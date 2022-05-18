# 软件工程学习笔记

在开发深度学习模型的过程中，领导提出了开发机器学习平台的需求，所以就遇到了将多个开发好的模型集成到一个代码项目的需求，也遇到了将集成脏数据处理，主动学习等工具包集成到代码项目的需求。
一开始手足无措的我，意识到了，我需要学习python开发中代码工程的相关技巧

## 参考资料
[Python最佳实践指南 2018](https://learnku.com/docs/python-guide/2018)

## 项目模块
```text
.
├── Domain
├── Classifier
├── Ner
├── tools
│   ├── Cleanlab
│   ├── ActiveLearning
├── libs
└── connector
└── tests
```
### 运用python的logging实现代码的日志模块
### 定义模型参数
datatype.py
使用@dataclass装饰器些属性类，来定义模型内部参数
config.yml
使用yaml类型的文件，来定义模型从外部输入的参数，增强配置文件的可读性
`import yaml`
### 运行装饰器修改可复用函数的功能
`@wrapper`