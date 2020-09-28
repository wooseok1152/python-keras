# jena dataset pandas를 이용하여 불러오기
import pandas as pd
import numpy as np

jena_dataset = pd.read_csv("./jena_climate/jena_climate_2009_2016.csv")

# dataframe인 jena dataset을 list로 변환
jena_dataset_list = []
for column in jena_dataset.columns[1:]:
    
    li = jena_dataset[column].tolist() 
    jena_dataset_list.append(li) 
    
# list인 jena dataset을 numpy로 변환
jena_dataset_numpy = np.array(jena_dataset_list)
jena_dataset_numpy = jena_dataset_numpy.T
print(jena_dataset_numpy, "\n")
print(jena_dataset_numpy.shape, "\n")
print(len(jena_dataset_numpy),"\n")

# jena_dataset_numpy 정규화
# mean = jena_dataset_numpy[:200000].mean(axis = 0)
# jena_dataset_numpy = jena_dataset_numpy - mean
# std = jena_dataset_numpy[:200000].std(axis = 0)
# jena_dataset_numpy = jena_dataset_numpy / std

#시게열 inout_data(x)와 output_data(y)를 반환하는 제너리이터 함수
def generator(data, lookback, delay, min_index, max_index, shuffle = False, batch_size = 128, step = 6):
    
    if max_index is None:
        
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        
        if shuffle == True:
            
            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
            
        else:
            
            if i + batch_size >= max_index :
                
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i = i + len(rows)
        
        # samples는 배치 내 sample들을 의미함()
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            
            indices = range(rows[j] - lookback, rows[j], step)
            print(list(indices), len(list(indices)))
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            
        yield samples, targets

#훈련, 검증, 테스트 제너레이터 준비
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(jena_dataset_numpy, lookback = lookback, delay = delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
for i, j in train_gen:
    
    sample = i
    target = j
    print("sample :", sample, "\n")
    print(len(sample), "\n")
    print("target :", target, "\n")
    print(len(target), "\n")    
    break

