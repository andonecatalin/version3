
import psycopg2
import numpy as np
import torch

class Protected_execution:
    def __init__(self):
        self.garbage='nothing'
    def create_query(name:str, table_name:str,add=False):
        if add:
            replacement_map=f'{name}_Close'
        else:
            replacement_map=[f'{name}_Open',f'{name}_High',f'{name}_Low',f'{name}_Close',f'{name}_Adj_Close',f'{name}_Volume,t']
        query=f"SELECT {', '.join(replacement_map)} FROM {table_name} WHERE {replacement_map[0]} IS NOT NULL ORDER BY t ASC"
        return query


    def insert_column(self,data,column_name,table_name,dbname,dbuser,dbpassword,dbhost,dbport):
        conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
        cursor=conn.cursor()
        query=f"""
            INSERT INTO {table_name} ({column_name})
            VALUES({[number for number in data]})
        """
        cursor.execute(query)
        conn.close()
    def create_cursor(dbname,dbuser,dbpassword,dbport,dbhost):
        conn=psycopg2.connect(dbname=dbname,user=dbuser,password=dbpassword,host=dbhost,port=dbport)
        cursor=conn.cursor
        return cursor,conn
    def fit_to_range_tensor(data, min_value=-5, max_value=5):
        """
        Fits a PyTorch tensor of numbers into a specified range.

        Args:
            data: The PyTorch tensor to fit.
            min_value: The minimum value in the desired range (inclusive).
            max_value: The maximum value in the desired range (inclusive).

        Returns:
            A new PyTorch tensor containing the fitted values.
        """
        if not data.numel():
            return torch.tensor([])  # Handle empty tensor

        # Find minimum and maximum values in the tensor
        data_min = torch.min(data)
        data_max = torch.max(data)

        # Handle constant data case (all elements have the same value)
        if data_min == data_max:
            return torch.full_like(data, min_value)

        # Calculate scaling factor
        scale = (max_value - min_value) / (data_max - data_min)

        # Fit the tensor using broadcasting
        fitted_data = min_value + scale * (data - data_min)

        # Clip values to the range (optional)
        clipped_data = torch.clamp(fitted_data, min_value, max_value)

        return clipped_data
    def tensor_shortner(tensor, requested_size):
        if isinstance(tensor, torch.Tensor):
            tensor_size=tensor.size()[0]
        elif isinstance(tensor, list):
            tensor_size=len(tensor)
        if tensor_size<requested_size:
            raise Exception("Requested size is bigger or equal to tensor")
        
        i=tensor_size
        while i!=0:
            if i % requested_size==0:
                return tensor[-i:]
            i-=1
    def image_builder(tensor:torch.Tensor, batch_size:int,shuffle=False):
        #makes consecutive batches in form of a list
        tensor_size=tensor.size()[0]
        batches=[]
        for i in range(tensor_size):
            if i+batch_size<tensor_size-1:
                batches.append([tensor[i:i+batch_size].tolist()])
        if shuffle:
            print("random function not yet implemented")
            exit()
            random.shuffle(batches)
            
        return batches
    
    
    def change(tensor:torch.Tensor, batch_size=32):
        #shape from image_builder function goes like this:
        #[batch number][leftover dimmension][element from batch]
        empty_list=[]
        for i in range(len(tensor)-1):
            first=tensor[i+1]
            last=tensor[i]
            diffrence=last-first
            average=(last+first)/2
            percent_diffrence=(diffrence/average)*100
            #limits to 20 points +/- to make training achivable
            if percent_diffrence >5:
                percent_diffrence=5
            elif percent_diffrence<-5:
                percent_diffrence=-5
            empty_list.append(percent_diffrence)
        return empty_list      


    def make_tensor(polars_fw,tensor, name, batch_size=32,result=False):
        #make it compatible with the tickers list
        name=name.lower()

        replacement_map=[f'{name}_low',f'{name}_high',f'{name}_volume']

        for name in replacement_map:
            #make it go trough all of the steps the original tensor goes
            concat=polars_fw[name]
            concat=torch.tensor(concat)
            concat=concat[1:]
            concat=Protected_execution.tensor_shortner(concat, batch_size)
            concat=Protected_execution.image_builder(concat, batch_size)
            if not result:
                concat=concat[:-1]
            concat=torch.tensor(concat)
            #move to the same device as original tensor
            concat=concat.to(str(tensor.device))
            #concatanate on dim=1
            tensor=torch.cat((tensor, concat),dim=1)
        return tensor

    def unix_to_time(timestamp):
        print("unix_to_time function not yet implemented")
        exit()
        dt_object=datetime.datetime.fromtimestamp(timestamp)
        while dt_object.weekday()>5:
            dt_object-=datetime.timedelta(days=1)
        time=dt_object.strftime('%Y-%m-%d')
        
        return time