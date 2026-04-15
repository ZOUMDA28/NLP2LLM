import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim,max_seq_len=5000):
        super().__init__()
        pe=torch.zeros(max_seq_len,dim)
        position=torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1),:]
        return x
if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    max_seq_len = 100
    
    # 初始化模块
    pe = PositionalEncoding(dim, max_seq_len)
    
    # 准备输入
    x = torch.zeros(batch_size, seq_len, dim) # 输入为0，直接观察PE值
    
    # 前向传播
    output = pe(x)
    
    # 验证输出
    print("--- PositionalEncoding Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
