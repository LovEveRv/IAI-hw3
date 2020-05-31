from dataset import SinaDataset

ds = SinaDataset('/Users/lovever/Documents/人工智能导论/实验三-情感分析/实验数据/sina/demo.json')
name, label, text = ds[0]
print(text.shape)