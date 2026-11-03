import torch
import torch.nn as nn





class TransformerAttention(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim*output_dim, last_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output
    





class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_num, predict_dim=1):
        super(TransformerMultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.query = nn.Linear(input_dim, input_dim * head_num)
        self.key = nn.Linear(input_dim, input_dim * head_num)
        self.value = nn.Linear(input_dim, input_dim * head_num)
        self.fc = nn.Linear(input_dim * head_num * output_dim, predict_dim)
        self.softmax = nn.Softmax(dim=1)

    def split_heads(self, tensor):
        # Split the last dimension into (head_num, new_last_dim)
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.head_num, tensor.size(-1) // self.head_num)
        # Transpose the result to (batch, head_num, new_last_dim, -1)
        return tensor.transpose(1, 2)
    

    def scaled_dot_product_attention(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention = self.softmax(scores)
        return torch.matmul(attention, v)

    def combine_heads(self, tensor):
        # Transpose and reshape the tensor to (batch, -1, head_num * new_last_dim)
        return tensor.transpose(1, 2).contiguous().view(tensor.size(0), -1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q, k, v = [self.split_heads(tensor) for tensor in (q, k, v)]
        attention = self.scaled_dot_product_attention(q, k, v)
        output = self.combine_heads(attention)
        output = self.fc(output)
        return output



class TransformerAttentionwithClassifierToken(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super(TransformerAttentionwithClassifierToken, self).__init__()
        self.input_dim = input_dim
        self.classifier_token = nn.Linear(input_dim, 1)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        classifier_token = torch.ones((x.shape[0], 1, self.input_dim)).to(x.device)*self.classifier_token.weight.data.squeeze(dim=-1)
        x = torch.cat([classifier_token.to(x.device), x], dim=-2)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        # using only the classifier token
        output = self.fc(output[:, 0, :])
        return output


class TransformerCrossAttention(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim):
        super(TransformerAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim*middle_dim, output_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.unsqueeze(dim=1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1)))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output





class TransformerAttentionwithPisition(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerAttentionwithPisition, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * token_num, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim]))

    def forward(self, x):
        seq_len = x.size(1)
        x_with_position = x + self.position_embeddings

        q = self.query(x_with_position)
        k = self.key(x_with_position)
        v = self.value(x_with_position)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output



class TransformerCrossAttentionwithPisition(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerCrossAttentionwithPisition, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim // 2, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim]))

    def forward(self, x):
        # first half to be q and second half to be k, v
        seq_len = x.size(1)
        x_with_position = x + self.position_embeddings

        q = self.query(x_with_position[:, :x_with_position.shape[1]//2, :])
        k = self.key(x_with_position[:, x_with_position.shape[1]//2:, :])
        v = self.value(x_with_position[:, x_with_position.shape[1]//2:, :])

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output


class TransformerAttentionwithCatPe(nn.Module):
    def __init__(self, input_dim, output_dim, last_dim=1, token_num=2):
        super(TransformerAttentionwithCatPe, self).__init__()
        input_dim = input_dim * 2
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim, last_dim)
        self.softmax = nn.Softmax(dim=1)
        self.position_embeddings = nn.Parameter(torch.zeros([token_num, input_dim//2]))

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.position_embeddings.expand(x.shape[0], -1, -1)
        x_with_position = torch.cat([x, pe], dim=-1)

        q = self.query(x_with_position)
        k = self.key(x_with_position)
        v = self.value(x_with_position)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

        attention = self.softmax(attention)
        output = torch.matmul(attention, v)

        output = output.view([output.shape[0], -1])
        output = self.fc(output)
        return output

