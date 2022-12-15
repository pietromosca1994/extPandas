from ast import Str
from tokenize import String
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from skimage.util import view_as_windows
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import plotly.express as px
from plotly.subplots import make_subplots

class extDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(extDataFrame, self).__init__(*args, **kwargs)

        self.scaler=None
    
        return None

    @property
    def _constructor(self):
        return type(self)

    def splitXY(self, X_features: List,
                        Y_features: List):
        '''Method for splitting a dataset in X and Y datasets

        :param List X_features: X features 
        :param List Y_features: Y features  
        
        :returns: X: train dataset
                    Y: Y train dataset

        :rtype: pextpdDataFrame
        '''

        X=self.get(X_features)
        Y=self.get(Y_features)

        return X, Y

    def splitTrainTest(self, *args, **kwargs):
        '''Method for splitting a dataset in X and Y datasets

        :param *arrays
        :param test_size    
        :param train_size 
        :param random_state
        :param shuffle 
        :param stratify 

        :returns: train_df: training dataframe
                    test_df: test dataframe 

        
        :rtype: extpandas.DataFrame
        '''
        train_df, test_df = train_test_split(self, *args, **kwargs)

        return train_df, test_df

    def Transform(self, transformation: String=None,
                        scaler=None,
                        *args, 
                        **kwargs):
        '''Method for transforming a dataset

        :param String transformation: transfomation 
        :param sklearn.preprovcessing.scaler scaler: scaler  

        :returns: scaler
       
        :rtype: sklearn.preprocessing.scaler
        '''   
        if scaler==None:
            if transformation=='StandardScaler': 
                scaler=preprocessing.StandardScaler(*args, **kwargs)
            
            elif transformation=='Normalizer':
                scaler=preprocessing.Normalizer(*args, **kwargs)
            
            elif transformation=='MinMaxScaler':
                scaler=preprocessing.MinMaxScaler(*args, **kwargs)  
            
            elif transformation=='RobustScaler':
                scaler=preprocessing.RobustScaler(*args, **kwargs)
            
            elif transformation=='TfidfVectorizer':
                scaler=text.TfidfVectorizer(*args, **kwargs)    

            scaler.fit(self.values)
        
        self.__init__(data=scaler.transform(self.values), columns=self.columns)
        self.scaler=scaler
        
        return None

    def InverseTransform(self,
                            scaler=None,
                             *args, **kwargs):
        '''Method for inverse transforming a dataset
        '''
        if scaler==None:
            scaler=self.scaler
            
        self.__init__(scaler.inverse_transform(self.values), columns=self.columns)
        self.scaler=scaler
        
        return None

    def getCorrMatrix(self, *args, **kwargs):
        '''Method for inverse transforming a dataset.
        The Pearson correlation coefficient [1] measures the linear relationship between two datasets. 
        The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between two datasets. 
        Kendall’s tau is a measure of the correspondence between two rankings. 

        :returns: CorrMatrix
       
        :rtype: extpandas.DataFrame
        '''
        # Pearson Correlation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
        # The Pearson correlation coefficient [1] measures the linear relationship between two datasets. 
        # Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. 
        # Correlations of -1 or +1 imply an exact linear relationship. 
        # Positive correlations imply that as x increases, so does y. 
        # Negative correlations imply that as x increases, y decreases.
        pearson_CorrMatrix=np.round(self.corr(method='pearson', *args, **kwargs), 2)

        # Spearman Correlation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
        # The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between two datasets. 
        # Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed. 
        # Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. 
        # Correlations of -1 or +1 imply an exact monotonic relationship. Positive correlations imply that as x increases, so does y.
        #  Negative correlations imply that as x increases, y decreases.
        spearman_CorrMatrix=np.round(self.corr(method='spearman', *args, **kwargs), 2)

        # Kendall’s tau is a measure of the correspondence between two rankings. 
        # Values close to 1 indicate strong agreement, and values close to -1 indicate strong disagreement. 
        # This implements two variants of Kendall’s tau: tau-b (the default) and tau-c (also known as Stuart’s tau-c). 
        # These differ only in how they are normalized to lie within the range -1 to 1; the hypothesis tests (their p-values) are identical. 
        # Kendall’s original tau-a is not implemented separately because both tau-b and tau-c reduce to tau-a in the absence of ties.
        kendall_CorrMatrix=np.round(self.corr(method='kendall', *args, **kwargs), 2)

        fig1=px.imshow(pearson_CorrMatrix, text_auto=True, title="Pearson's Correlation")
        fig2=px.imshow(spearman_CorrMatrix, text_auto=True, title="Spearman's Correlation")
        fig3=px.imshow(kendall_CorrMatrix, text_auto=True, title="Kendall's Correlation")

        # plot figure
        fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True, subplot_titles=("Pearson's Correlation", "Spearman's Correlation", "Kendall's Correlation"))
        fig.add_trace(fig1['data'][0], row=1, col=1)
        fig.add_trace(fig2['data'][0], row=1, col=2)
        fig.add_trace(fig3['data'][0], row=1, col=3)
        
        return fig

    def ViewAsWindows(self,
                        window_length: int):
        '''Method for preprocessing timseries data 

        :param int window length: length (in samples) of the window
        
        :returns: data_as_windows_3D

        :rtype: numpy.ndarray
        '''

        data_as_windows_3D=np.squeeze(view_as_windows(self.values, (window_length, self.values.shape[1])))
        
        #shape=data_as_windows_3D.shape
        #data_as_windows=np.reshape(data_as_windows_3D, (shape[0]*shape[1], shape[2]))
        #self.__init__(data_as_windows, columns=self.columns)
        
        return data_as_windows_3D

    def plotDistribution(self,
                            nbins=50) -> plotly.graph_objs.Figure:
        '''Function to plot data distribution

        :returns: fig

        :rtype: plotly.graph_objs.Figure     
        '''
        fig = make_subplots(rows=len(self.columns), cols=1)

        for row in range(len(self.columns)):
            fig.add_trace(
                go.Histogram(x=self[self.columns[row]], nbinsx=nbins, name=self.columns[row]),
                row=row+1, col=1
            )
        #            
        # if len(self.columns)==1:
        #         fig.add_trace(
        #         go.Histogram(x=self[self.columns[0]], nbinsx=20, name=self.columns[0]),
        #         row=1, col=1
        #     )
        # else:
        #     for row in range(1, len(self.columns)):
        #         fig.add_trace(
        #             go.Histogram(x=self[self.columns[row-1]], nbinsx=20, name=self.columns[row-1]),
        #             row=row, col=1
        #         )
        
        return fig

    def removeOutliers(self, columns=None, method='IQR', *args, **kwargs):
        ''' Method to remove outliers from data 

        '''
        if columns==None:
            columns=self.columns

        if method=='IQR':
            self.loc[:, columns]=self.IQR_method(data=self.loc[:, columns], *args, **kwargs)
      
        return None

    def IQR_method(self, 
                    data,
                    lo_quantile: float =10,
                    hi_quantile: float =90):
        ''' InterQuantile Range Method
        '''
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):  # check for numerical data 
                Qlow = np.percentile(data[col], lo_quantile)
                Qhigh = np.percentile(data[col], hi_quantile)
                IQR = Qhigh - Qlow
                outliers_count=sum(data[col] <= Qlow - 1.5 * IQR)+sum(data[col] >= Qhigh + 1.5 * IQR) # outliers count 
                outliers_perc=round(outliers_count/data.shape[0]*100, 2)    # outliers percentage        
                print('[INFO] Found ', outliers_count, ' (', outliers_perc, '%) outliers in ', col)
                
                # data.iloc[data[col].values <= Qlow - 1.5 * IQR, data.columns.get_loc(col)]= Qlow
                # data.iloc[data[col].values >= Qhigh + 1.5 * IQR, data.columns.get_loc(col)]= Qhigh

                data.loc[:, col].mask(data[col] <= Qlow - 1.5 * IQR, Qlow, inplace=True)
                data.loc[:, col].mask(data[col] >= Qhigh + 1.5 * IQR, Qhigh, inplace=True)
        
        return data 

    def NARX(self, ex_order: List = None,
                   ex_delay: List = None,
                   auto_order: int = None,
                   pred_step: int = None,
                   inplace: bool =False):
        '''Method for processing the data in a NARX fashion.
            y(t+k)=f(y(t), ..., y(t-p+1), 
                    x1(t-d1), ..., x1(t-d1-q1+1), ...
                    xm(t-dm), ..., xm(t-dm-qm+1), e(t))
            Where:
                q (ex_order): exogenous order 
                d (ex_delay): exogenous delay 
                p (auto_order): auto order
                k (pred_step): prediction step
        '''
        NARX=pd.DataFrame()      
       
        for d, column in enumerate(self.columns):
            #self[column]=self.columns.shift(ex_delay[d])
            index=[]
            for i in range(ex_order[d]+1):
                index.append(column+'_d'+str(ex_delay[d])+'_o'+str(i))
            #NARX_row=pd.concat([NARX_row, pd.DataFrame(self[column].shift(ex_delay[d]).values[:ex_order[d]], index=index)], axis=1)

            NARX=pd.concat([NARX, pd.DataFrame(view_as_windows(np.array(self[column].shift(ex_delay[d])), ex_order[d]+1), columns=index)], axis=1)
        
        # drop NaNM due to delay 
        NARX.dropna(inplace=True)

        if inplace==True:
            self.__init__(data=NARX.values, columns=NARX.columns)
            return None
        else:
            return NARX

        