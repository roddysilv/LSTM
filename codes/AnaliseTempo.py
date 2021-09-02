import pandas as pd

def Save(house,rscore,loss,val_loss,time):
    try:
        df = pd.read_csv('History_Residential.csv')
    except:
        df = pd.DataFrame(columns = ['Residential', 'R2Score','Loss','Val_Loss','Time'])
        
    aux = pd.DataFrame([[house,rscore,loss,val_loss,time]],columns = ['Residential', 'R2Score','Loss','Val_Loss','Time'])
    
    df = df.append(aux)

    df.to_csv('History_Residential.csv',index=False)