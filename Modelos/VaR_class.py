import pandas as pd
import numpy as np
from scipy import stats
from arch import arch_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from scipy import optimize

class VaR(object):
    
    def __init__(self,retornos,significancia):
        
        self.retornos=retornos.dropna()
        self.indice=retornos.index
        self.significancia=significancia
        
    def historico_normal(self,train,test,in_sample=False):
    
        '''
        Calcula el VaR asumiendo un distribución normal.

        params:

        train -> Series object with date index.
        test -> Series object with date index.
        confianza -> float or list of float.

        returns:

        VaR -> Series object with date index.
        '''
        if in_sample==True:
            train=pd.concat([train,test])
            indice_test=train.index
            
        indice_train=train.index
        indice_test=test.index
        significancia=self.significancia
        Media=train.mean() ## Mean
        Sd=train.std() ## Variance
        VaR=pd.DataFrame(np.repeat(stats.norm.ppf(significancia,loc=Media,scale=Sd),len(test)),columns=[f'VaR_{significancia}'])
        return VaR
        
    def garch(self,train,test,in_sample=False,p=1,q=1):
        
        '''
        Calcula el VaR asumiendo un distribución normal pero con una varianza que sigue un proceso GARCH.

        params:

        train -> Series object with date index.
        test -> Series object with date index.
        confianza -> float or list of float.

        returns:

        VaR -> Series object with date index.
        '''
        
        if in_sample==True:
            
            rt=train.copy()
        else:
            rt=pd.concat([train,test])
        
        am = arch_model(rt, vol="Garch", p=p,o=0, q=q,dist="skewt")
        res = am.fit(disp="off", last_obs=train.index[-1])
        forecasts = res.forecast(start=test.index[0], reindex=False)

        cond_mean = forecasts.mean
        cond_var = forecasts.variance
        
        q = am.distribution.ppf([self.significancia], res.params[-2:])
        value_at_risk = -(cond_mean.values - np.sqrt(cond_var).values * q[None, :])
        value_at_risk = pd.DataFrame(value_at_risk, columns=[f'VaR_{self.significancia}'])
        return value_at_risk
    
    
    ### Caviar-SAV
    
    def caviar_sav(self,train,test,in_sample=False):
        
        ### Definimos datos iniciales.
        pval=self.significancia
        obs_train=len(train)
        obs_test=len(test)
        emp_quantile=np.quantile(train,pval)
        VaR=np.repeat(0,obs_train).astype('float') ### Matrix for fill
        VaR_test=np.repeat(0,obs_test).astype('float')
        
        
        ### funcion
        
        def quantile_reg(beta,y_dep,emp_quantile,VaR,n_y_dep,predict,h_test):
            VaR[0]=-1*emp_quantile
            ## itera
            if not predict:
                for i in range(1,n_y_dep):
                    VaR[i] = beta[0]+beta[1]*VaR[i-1]+beta[2]*abs(y_dep[i-1])
            if predict:
                Var_test=np.repeat(0,obs_test).astype('float')
                Var_test[0]=VaR[n_y_dep-1]
                VaR=VaR_test.copy()
                for j in range(1,obs_test):
                    VaR[j] = beta[0]+beta[1]*VaR[j-1]+beta[2]*abs(y_dep[j-1])   
            return VaR 
            
            
        ### Pinball
        def hit(data,VaR,pval):
            Hit=(data<-VaR)-pval
            RQ=np.dot(-1*np.transpose(Hit),data+VaR)
            return RQ if np.isfinite(RQ) else 1e+100
        ## Objetive function

        def fun_obj(beta,y_dep=train.iloc[:,0].to_numpy(),emp_quantile=emp_quantile,VaR=VaR,n_y_dep=obs_train,predict=False,pval=self.significancia,h_test=1):
            return hit(y_dep,quantile_reg(beta,y_dep,emp_quantile,VaR,obs_train,False,h_test=1),pval)

        ## Matrices de betas 
        beta_inicial=np.random.uniform(size=[10000,3])

        ## A cada fila le aplicamos el cavias model
        VaR_guest=np.atleast_2d(np.apply_along_axis(lambda x:hit(train.iloc[:,0].to_numpy(),quantile_reg(x,train.iloc[:,0].to_numpy(),emp_quantile,VaR,obs_train,False,h_test=1),self.significancia),1,beta_inicial))
        df_resultados=pd.DataFrame(np.append(VaR_guest.T,np.array(beta_inicial),axis=1),columns=['R','B0','B1','B2'])\
        .sort_values('R')\
        .iloc[0:10,:] ### Seleccionamos las mejores

        ### Optimizador.
        Resultados=pd.DataFrame({'Fun_eval':np.repeat(0,len(df_resultados.R)),
                                 'B0':np.repeat(0,len(df_resultados.R)),
                                 'B1':np.repeat(0,len(df_resultados.R)),
                                 'B2':np.repeat(0,len(df_resultados.R))})

        i=0
        for beta in zip(df_resultados.B0,df_resultados.B1,df_resultados.B2):
            R=optimize.minimize(fun_obj,beta,method='Nelder-Mead', bounds=[(-np.inf,np.inf) for x in range(1,4)])
            Resultados.loc[i,'Fun_eval']=R.fun
            Resultados.loc[i,'B0']=R.x[0]
            Resultados.loc[i,'B1']=R.x[1]
            Resultados.loc[i,'B2']=R.x[2]
            i=i+1
        Resultados.sort_values('Fun_eval',inplace=True)

        ### optimal betas
        betas=Resultados.iloc[0,:].drop('Fun_eval').to_numpy()
        ### out-sample result
        df_outsample=quantile_reg(betas,test.iloc[:,0].to_numpy(),emp_quantile,VaR,obs_train,predict=True,h_test=obs_test)

        return pd.DataFrame({f'VaR_{self.significancia}':-df_outsample},index=test.index)
    
    
    
    ### QCNN
    
    def qcnn(self,train,test,in_sample=False,nlag=3): 
        
        indice_train=max(train.index)
        indice_test=min(test.index)
        if in_sample==False:
            returns=pd.concat([train,test]).copy()
        else:
            returns=train.copy()
        returns.columns=['RT']
        for i in range(1,nlag+1):            
            returns[f'LAG{i}']=returns.RT.shift(i)
            returns.dropna(inplace=True)
        if in_sample==True:    
            returns_train=returns.copy()
            returns_test=returns.copy()
        else:
            returns_train=returns[returns.index<=indice_train]
            returns_test=returns[returns.index>=indice_test]
            
        X_train=returns_train.copy().drop('RT',axis=1).to_numpy()
        X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        Y_train=returns_train.copy().RT.to_numpy()            
        X_test=returns_test.copy().drop('RT',axis=1).to_numpy()
        X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1))
        Y_test=returns_test.copy().RT.to_numpy()  

        def quantile_loss(q,y_true,y_pred):

            """
            q -- quantile level
            y_true -- true values
            y_pred -- predicted values

            """
            diff = (y_true - y_pred)
            mask = y_true >= y_pred
            mask_ = y_true < y_pred
            loss = (q * K.sum(tf.boolean_mask(diff, mask), axis=-1) - (1 - q) * K.sum(tf.boolean_mask(diff, mask_), axis=-1))
            return loss

        def modelo(trainX, trainY,p_value):
            """
            trainX -- input values; shape: [number of samples, NUM_UNROLLINGS, 1]
            trainY -- output values (inputs shifted by 1); shape: [number of samples, NUM_UNROLLINGS, 1]
            """
            np.random.seed(123)
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss=lambda y_true,y_pred: quantile_loss(p_value,y_true,y_pred))
            model.fit(X_train, Y_train,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,shuffle=True,verbose=0)
            return model
        
        q_05=modelo(X_train,Y_train,self.significancia).predict(X_test)
        return pd.DataFrame(q_05,columns=[f'VaR_{self.significancia}'])