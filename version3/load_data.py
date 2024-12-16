import support_functions
import polars as pl
import torch
import data_download
from torch.utils.data import DataLoader, TensorDataset

class give_data:
    def __init__(self):
        self.dbname='ai_dataset'
        self.dbuser='postgres'
        self.dbpassword='parola'
        self.dbport=5432
        self.dbhost='127.0.0.1'
        self.table_name='tabela'


        self.tickers = ["AAPL", "VIX.INDX", 'SPY', 'XAUUSD.OANDA', 'BCO.ICMTRADER','NVDA','BTC.AGR']
        for i in range(len(self.tickers)):
            self.tickers[i]=self.tickers[i].replace('.','_')
        self.net_functions=support_functions.Protected_execution


    def l2_normalize(self,tensor, dim=0, epsilon=1e-12):
            """
            Applies L2 normalization to a tensor along the specified dimension.

            Args:
            tensor (torch.Tensor): The input tensor to normalize.
            dim (int): The dimension along which to normalize. Default is 0.
            epsilon (float): A small value to avoid division by zero. Default is 1e-12.

            Returns:
            torch.Tensor: The L2-normalized tensor.
            """
            # Calculate the L2 norm along the specified dimension, keeping dimensions
            l2_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
            
            # Prevent division by zero by adding a small value epsilon to the norm
            l2_norm = torch.clamp(l2_norm, min=epsilon)
            
            # Divide the tensor by its L2 norm
            normalized_tensor = tensor / l2_norm
            
            return normalized_tensor
    def get_data(self,eval=False):
        
        query=self.net_functions.create_query(self.tickers[0],self.table_name)
        cursor,conn=self.net_functions.create_cursor(self.dbname,self.dbuser,self.dbpassword,self.dbport,self.dbhost)
        data=pl.read_database(query=query,connection=conn)
        data=data.drop_nulls()
        conn.close()

        if eval:
            train_data=data['aapl_close']
            labels=self.net_functions.change(train_data)
            train_data=torch.tensor(train_data)
            labels=torch.Tensor(labels)
            train_data=train_data[1:]
            batch_size=32
            shortened_tensor=self.net_functions.tensor_shortner(train_data, batch_size)
            image_tensor=self.net_functions.image_builder(shortened_tensor,batch_size=batch_size)
            labels=self.net_functions.tensor_shortner(labels,batch_size)
            labels=[int(tensor.item()) for tensor in labels]
            image_tensor=torch.tensor(image_tensor)
            image_tensor=self.net_functions.make_tensor(data,image_tensor,self.tickers[0],result=True)
            labels=[x+5 for x in labels]
            labels=torch.tensor(labels,dtype=torch.float32)
            temp_labels=self.net_functions.image_builder(labels,32)
            temp_labels=torch.Tensor(temp_labels)
            image_tensor=torch.cat((image_tensor,temp_labels),dim=1)
            for i in range(len(image_tensor[0].size())):
                image_tensor[i]=self.l2_normalize(image_tensor[i])
            final_input=image_tensor[-1,:,:]
            return final_input


        train_data=data['aapl_close']
        print(data)
        labels=self.net_functions.change(train_data)
        train_data=torch.tensor(train_data)
        labels=torch.Tensor(labels)
        train_data=train_data[:-1]
        batch_size=32
        shortened_tensor=self.net_functions.tensor_shortner(train_data, batch_size)
        image_tensor=self.net_functions.image_builder(shortened_tensor,batch_size=batch_size)
        image_tensor=image_tensor[:-1]
        labels=torch.Tensor(labels)
        labels=self.net_functions.tensor_shortner(labels,batch_size)
        labels=[x+5 for x in labels]
        labels=torch.tensor(labels,dtype=torch.float32)
        image_tensor=torch.tensor(image_tensor)
        image_tensor=self.net_functions.make_tensor(data,image_tensor,self.tickers[0])
        temp_labels=self.net_functions.image_builder(labels[:-1],32)
        temp_labels=torch.Tensor(temp_labels)
        image_tensor=torch.cat((image_tensor,temp_labels),dim=1)
        labels=labels[batch_size+2:]
        labels=labels.float()
        for i in range(len(image_tensor[0].size())):
            image_tensor[i]=self.l2_normalize(image_tensor[i])
        train_dataset = TensorDataset(image_tensor, labels)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        return train_loader