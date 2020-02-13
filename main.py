import streamlit as st
import pandas as pd 
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go 
from datetime import timedelta  
from sklearn.ensemble import IsolationForest
from fbprophet import Prophet

filename = 'datatest.txt'


# ======== Display function ===========
def run_app(filename):
	""" Initial function that starts all the interface

	This initial function display widget to identify if it is a timeseries function or not
	Depending on the choice, a different function is triggered

	Args:
		filename (string): path of input csv file for the application

	Returns:
		None
	"""
	df = load_data(filename)
	columns = df.columns.values.tolist()
	st.sidebar.header('Data definition')
	time_data = st.sidebar.checkbox('is dataset a timeseries dataset ?')
	if time_data :
		time_series_data(df, columns)
	else :
		non_time_series_data(df,columns)



def non_time_series_data(df,columns): 
	""" Display the widget and vizs for a non-time-series dataframe 
	
	This function calls for the widgets, the visualisation and the anomaly detection function

	Args:
		df (dataframe): Pandas dataframe of the input data
		columns (list): List of the dataframe columns name  
	Returns: 
		None
	"""
	# Widgets (sidebar)
	field_to_ignore = fields_to_ignore(columns)
	filtered_fields = list(set(columns) - set(field_to_ignore))
	discri_field = discriminator_field(filtered_fields)

	# Visualisation 
	data_summary_display(filtered_fields, df, discri_field)

	# Anomaly detection 
	algorithm_detection_normal(df,filtered_fields)

	
def time_series_data(df, columns):
	""" Display the widget and vizs for a time-series dataframe 
	
	This function calls for the widgets, the visualisation and the anomaly detection function

	Args:
		df (dataframe): Pandas dataframe of the input data
		columns (list): List of the dataframe columns name  
	Returns: 
		None

	"""
	# Pick time field
	
	time_field = st.sidebar.selectbox(label= "Choose the time series field ?",
						options = ['None'] + columns,
						index = 0)


	# Trigger flow when time field is choose	
	if time_field != 'None':
		no_time_columns = list(set(columns) - set([time_field]))
		field_to_ignore = fields_to_ignore(no_time_columns)
		filtered_fields = list(set(no_time_columns) - set(field_to_ignore))
		discri_field = discriminator_field(filtered_fields)

		data_summary_display(filtered_fields, df, discri_field, time_field)
		df_2, df_2_summary = data_summary(df, time_field, filtered_fields, discri_field)
		time_series_graph(df_2, time_field, filtered_fields)


def fields_to_ignore(fields):
	""" Display  widget of field to be ignore in analysis 

	Args : 
		fields (list) : List of the dataframe columns name minus time field, if time series

	Return :
		fields_to_ignore (list): List of field to be ignored for analysis
	
	"""
	return st.sidebar.multiselect(label = "Do you want to ignore some field ?", 
						options = ['None'] + fields)

def discriminator_field(fields):
	""" Display widget to pick one discriminator field

	Discrimantor field should be categorical. It can be a label (y_true) of the data
	
	Args : 
		fields (list): Dataframe columns minus time field (if time series) and fields to ignore  
	
	Returns : 
		discriminator_field (String): Field pick as discriminator. 'None' if nothing chosen

	""" 
	discri_field = 'None'
	discri = st.sidebar.checkbox('Categorical field')
	# filtered_fields = list(set(no_time_columns) - set(field_to_ignore))
		
	if discri : 
		discri_field = st.sidebar.selectbox(label= "Choose the categorical field ?",
					options = np.insert(fields,0,'None'),
					index = 0)

	else :  
		discri_field = 'None'

	return discri_field

@st.cache
def load_data(filename):
	""" Load csv file in dataframe

	Function is cached to avoid the reload at each interaction

	Args: 
		filename (string): Path to the data .csv file

	Returns:
		df (DataFrame): Initial dataframe

	"""
	return pd.read_csv(filename)
	
def data_summary_display(filtered_fields : list, df, discri_field = 'None', time_field = 'None'):
	""" Display the data table for summary and data statistics 

	Data statistics include  mean, max, etc ... 
	Function used for both time and non time series

	Args: 
		fieldtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  
		df (DataFrame): initial dataframe
		discri_field (String): discriminator field
		time_field (String): time field
	Returns: 
		None

	"""
	st.title('Data Summary')
	st.header('Data table') 
	df_2, df_2_summary = data_summary(df, time_field, filtered_fields, discri_field)
	if discri_field != 'None':
		st.dataframe(df_2.style.apply(background_color_col, axis=0, subset = [discri_field]))
		st.header('Data description - Summary')  
		st.dataframe(df_2.describe().style.apply(background_color_col, axis=0, subset = [discri_field]))
	else : 
		st.dataframe(df_2)
		st.header('Data description - Summary')
		st.dataframe(df_2.describe())
	data_distribution_graph(filtered_fields, df_2, discri_field)
	


def data_distribution_graph(filtered_fields, df_2, discri_field = 'None'): 
	""" Display distribution graph 

	Args: 
		fieldtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  
		df_2 (DataFrame): Dataframe reprocessed (see data_summary function)
		discri_field (String): discriminator field
	Returns: 
		None

	"""
	st.sidebar.header('Distribution graph')
	plot_field = st.sidebar.selectbox(label= "Choose the field to plot ?",
						options = ['None'] + filtered_fields,
						index = 0)
	bin_slider = st.sidebar.slider(label = 'bin_size', min_value = 0.01, max_value = 5.0
		, value = 1.0, step = 0.05)
	df_corr = calculate_correlation_matrix(df_2)
	if plot_field != 'None': 
		if discri_field == 'None':
			st.header(f'Distribution graph for {plot_field} (no group)')
			fig = ff.create_distplot([df_2[plot_field].values], ['No Category'], bin_size=bin_slider)
			st.plotly_chart(fig)
		else :
			group_list = df_2[discri_field].unique()
			if len(group_list) > 10 : 
				st.header(f'Distribution graph for {plot_field} (no group)')
				st.error('Not a categorical field\n More than 10 categories\n Please revise categorical field choice')
				fig = ff.create_distplot([df_2[plot_field].values], ['No Category'], bin_size=bin_slider)
				st.plotly_chart(fig)
			else : 
				st.header(f'Distribution graph for {plot_field} - grouped on field {discri_field}')
				df_group = [df_2[df_2[discri_field]  ==  i][plot_field] for i in group_list]
				fig = ff.create_distplot(df_group, [str(el) for el in group_list], bin_size=bin_slider)
				st.plotly_chart(fig)
	st.header('Correlation heatmap')
	
	fig = go.Figure(data=go.Heatmap(
                   z=df_corr.values,
                   x=df_corr.columns,
                   y=df_corr.columns,
                   colorscale = 'RdBu',
                   reversescale = True, 
                   zmax = 1,
                   zmin = -1 ))
	st.plotly_chart(fig)


def time_series_graph(df2, time_field, filtered_fields):
	""" Display time graph 
	
	For time series only
	Includes widgets to select the target feature and will automatically display a time-plot graph 
	
	Args: 
		df_2 (DataFrame): Dataframe reprocessed (see data_summary function)
		fieldtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  
		time_field (String): time field
	Returns: 
		None

	"""
	st.sidebar.header('Time graph')
	plot_fields =  st.sidebar.multiselect(label = "Choose the field to plot ?", 
						options = filtered_fields)
	if len(plot_fields) > 0 : 
		st.header(f'Time graph')
		for plot_field in plot_fields : 
			st.write(plot_field)
			fig = go.Figure([(go.Scatter(x=df2[time_field], y=df2[plot_field], name = plot_field))])
			st.plotly_chart(fig)

	algorithm_detection_time_series(df2, time_field, filtered_fields)

def algorithm_detection_normal(df2, filtered_fields):
	""" Menu for anomaly detection choice for non-time series data

	Include widget for choosing the algorithm and the feature
	Display the result via a plot

	Args: 
		fieldtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  
		df_2 (DataFrame): Dataframe reprocessed (see data_summary function)

	Returns: 
		None
	"""
	st.sidebar.header('Algorithms')
	algo = st.sidebar.selectbox(label= "Choose the algorithm to use ?",
						options = ['None','IsolationForest'],
						index = 0)
	feature = st.sidebar.selectbox(label= "Choose the feature to use ?",
						options = ['None'] + filtered_fields,
						index = 0)
	if feature != 'None'and algo != 'None':
		if algo == 'IsolationForest' : 
			isolation_forest_plot_normal(df2, filtered_fields, feature)
		else : 
			pass 
	

def algorithm_detection_time_series(df2, time_field, filtered_fields): 
	""" Menu for anomaly detection choice for time series data

	Include widget for choosing the algorithm and the feature
	Display the result via a plot

	Args: 
		df_2 (DataFrame): Dataframe reprocessed (see data_summary function)
		time_field (String): time field
		filtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  

	Returns: 
		None
	"""
	st.sidebar.header('Algorithms')
	algo = st.sidebar.selectbox(label= "Choose the algorithm to use ?",
						options = ['None','IsolationForest','Prophet'],
						index = 0)
	feature = st.sidebar.selectbox(label= "Choose the feature to use ?",
						options = ['None'] + filtered_fields,
						index = 0)
	
	if feature != 'None'and algo != 'None':
		if algo == 'IsolationForest' : 
			isolation_forest_plot_ts(df2, filtered_fields, feature, time_field)
		elif algo == 'Prophet': 
			prophet_plot(df2, filtered_fields, feature, time_field)

		

# Funtion  utilities

@st.cache
def data_summary(df, time_field, filtered_fields, discri_field):
	if time_field == 'None': 
		df_2 = df[filtered_fields]
	else :
		df_2 = df[[time_field] + filtered_fields].astype({time_field:'datetime64'})
	df_2_summary = df_2.describe()
	return df_2, df_2_summary


@st.cache
def calculate_correlation_matrix(df):
	return df.corr()

def background_color_col(col, color = '#DA70D6'):
	return [f'background-color: {color}' for i in col]


def isolation_forest(df_value, anomaly_rate):
	""" Run isolation Forest algorithm

	Args: 
		df_value (Pandas Series): Preselected column to run the isolation forest algorithm
		anomaly-rate (float): anomaly ratio expected in the algorithm 

	Returns: 
		df_anomaly (DataFrame): input dataframe enchanced with 'outlier' columns indicated if the value is an anomaly/outlier (1) or not(0) 
	
	"""
	X = df_value.values.reshape(-1,1)
	clf = IsolationForest(contamination = anomaly_rate / 100)
	clf.fit(X)
	y = clf.predict(X)
	return pd.concat([df_value, pd.DataFrame(y, columns = ['Outlier'])], axis= 1)

def prophet(df,time_field, feature, daily, weekly, yearly):
	""" Run prophet time prediction algorithm

	For time series scenario only

	Args: 
		df (DataFrame): dataframe with input data minus filtered columns
		time_field (String): time field
		feature (String): Selected column for the forecast
		daily (boolean): is time series daily periodic ? 
		weekly (boolean): is time series weekly periodic ? 
		yearly (boolean): is time series yearly periodic ? 

	Returns: 
		forecast (DataFrame): forecast value for the next 3000 period range
	
	"""
	df_prophet = df[[time_field,feature]]
	df_prophet.columns = ['ds','y']
	df_prophet.set_index('ds')
	ml = Prophet(growth = 'linear', daily_seasonality = daily, weekly_seasonality = weekly,
		yearly_seasonality = yearly, changepoint_prior_scale = 0.001)
	ml.fit(df_prophet)
	period_range = 3000
	future = ml.make_future_dataframe(periods = period_range, freq = '1min')
	forecast = ml.predict(future)
	return forecast.iloc[-period_range-1:]

def prophet_plot(df, filtered_fields, feature, time_field): 
	""" Plot prophet forecast with past value and forecast

	Plot includes the incertitude range in the forecast 

	Args: 
		df (DataFrame): dataframe with input data minus filtered columns
		time_field (String): time field
		feature (String): Selected column for the forecast
		filtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  

	Return: 
		None

	"""
	st.header(f'Prophet time prediction on {feature} field')
	fig = go.Figure()
	# Settings 
	daily = st.sidebar.checkbox('Daily pattern ?')
	weekly = st.sidebar.checkbox('Weekly pattern ?')
	yearly = st.sidebar.checkbox('Yearly pattern ?') 


	df_forecast = prophet(df, time_field, feature, daily, weekly, yearly)


	fig.add_trace(go.Scatter(x= df[time_field],y= df[feature], 
									mode='lines', name = 'past'))
	fig.add_trace(go.Scatter(x = df_forecast['ds'], y = df_forecast['yhat'],
									name = 'prediction'))


	fig.add_trace(go.Scatter(
	    x=pd.concat([df_forecast['ds'], df_forecast['ds'].iloc[::-1]]),
	    y=pd.concat([df_forecast['yhat_upper'], df_forecast['yhat_lower'].iloc[::-1]]),
	    fill='toself',
	    fillcolor='rgba(220,20,60,0.2)',
	    line_color='rgba(255,255,255,0)',
	    showlegend=False,
	    name='incertitude',
	))

	fig.update_layout(
		title = f"{feature} through time with prediction", 
		xaxis_title = f"{time_field}",
		yaxis_title = f"{feature}")
	st.plotly_chart(fig)



def isolation_forest_plot_ts(df, filtered_fields, feature, time_field):
	""" Plot isolation forest results on a time axis

	Plot for time series scenario
	Plot display in red the range where anomaly/outlier is detected 

	Args: 
		df (DataFrame): dataframe with input data minus filtered columns
		time_field (String): time field
		feature (String): Selected column for the forecast
		filtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  

	Return: 
		None

	"""
	st.header(f'Isolation Forest on {feature} field')
	anomaly_rate = st.sidebar.slider(label = 'Target Anomaly Rate', min_value = 0.5, max_value = 50.0
		, value = 10.0, step = 0.5)
	df_result0 = isolation_forest(df[feature], anomaly_rate)
	df_result = pd.concat([df[time_field], df_result0], axis= 1)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x= df_result[time_field],y= df_result[feature], mode = 'lines'))
	ranges = define_range_index(df_result[df_result['Outlier'] == -1].index.values)
	shapes = define_shapes_ts(ranges, df_result, time_field)
	
	fig.update_layout(
		title = f"{feature} through time", 
		xaxis_title = f"{time_field}",
		yaxis_title = f"{feature}",
		shapes=shapes)
	st.plotly_chart(fig)

def isolation_forest_plot_normal(df, filtered_fields, feature):
	""" Plot isolation forest results

	Plot for non time series scenario
	Values are ordered and place in increasing order on the  x axis
	Plot display in red the range where anomaly/outlier is detected 

	Args: 
		df (DataFrame): dataframe with input data minus filtered columns
		feature (String): Selected column for the forecast
		filtered_fields (list): Dataframe columns minus time field (if time series) and fields to ignore  

	Return: 
		None

	"""
	st.header(f'Isolation Forest on {feature} field')
	anomaly_rate = st.sidebar.slider(label = 'Target Anomaly Rate', min_value = 0.5, max_value = 50.0
		, value = 10.0, step = 0.5)
	df_result = isolation_forest(df[feature], anomaly_rate)
	df_result.sort_values([feature], ascending = True, inplace=True)
	df_result.reset_index(inplace = True)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x= df_result[feature],y= df_result.index, mode = 'markers'))
	ranges = define_range_index(df_result[df_result['Outlier'] == -1].index.values)
	shapes = define_shapes(ranges, df_result, feature)
	
	fig.update_layout(
		title = f"{feature} sorted by values",
		xaxis_title= f"{feature}",
    	yaxis_title="index #(sort by values)",
    	shapes=shapes)
	st.plotly_chart(fig)

def define_shapes_ts(ranges, df, x_feature): 
	""" Define rectangle with display the anomaly area

	Function for time series plot 

	Args: 
		ranges (list): list of list which are the indicate the anomaly ranges on the x axis
		df (DataFrame): dataframe with input data minus filtered columns
		x_feature (String): name of the x_feature (in our time series case, the time field)

	Return: 
		shapes (plotly.layout.shape): Shape that define the anomaly/outlier area

	"""
	shapes = []
	for r in ranges :
		x0 = df[x_feature].iloc[r[0]]
		x1 = df[x_feature].iloc[r[1]]
		el = [go.layout.Shape(
					type='rect',
					xref ='x',
					yref = 'paper',
					x0=x0, 
					y0=0,
					x1=x1,
					y1=1, 
					fillcolor="LightSalmon",
	            	opacity=0.5,
	            	layer="below",
	            	line_width=0,
					)]
		shapes += el
	return shapes

def define_shapes(ranges, df, x_feature):
	""" Define rectangle with display the anomaly area

	Function for non-time series plot 

	Args: 
		ranges (list): list of list which are the indicate the anomaly ranges on the x axis
		df (DataFrame): dataframe with input data minus filtered columns
		x_feature (String): name of the x_feature

	Return: 
		shapes (plotly.layout.shape): Shape that define the anomaly/outlier area

	""" 
	shapes = []
	for r in ranges :
		if r[0] == 0 : 
			x0 = df[x_feature].iloc[0]
		else : 
			x0 = (0.8*df[x_feature].iloc[r[0]] + 0.2*df[x_feature].iloc[r[0] - 1]) 
		if r[1] == len(df) - 1 : 
			x1 = df[x_feature].iloc[-1]
		else : 
			x1 = (0.8*df[x_feature].iloc[r[1]] + 0.2*df[x_feature].iloc[r[1] + 1]) 
		el = [go.layout.Shape(
					type='rect',
					xref ='x',
					yref = 'paper',
					x0=x0, 
					y0=0,
					x1=x1,
					y1=1, 
					fillcolor="LightSalmon",
	            	opacity=0.5,
	            	layer="below",
	            	line_width=0,
					)]
		shapes += el
	return shapes


def define_range_index(index): 
	""" Calculate the range value of the filtered index

	Index of the detected anomaly/outlier
	Function reduced this index in ranges
	Exemple : [1,2,3,4,6,7,8,10] => [1,4],[6,8],[10] 

	Args: 
		index (list): list of index corresponding to an anomaly

	Returns: 
		ranges (list<list>) : list of list, corresponding to the ranges
	"""
	ranges = []
	min_range = None
	for el in index : 
	    if min_range == None: 
	        min_range = el
	        previous_el = el
	    elif el == previous_el + 1: 
	        previous_el = el 
	    else : 
	        max_range = previous_el 
	        ranges += [[min_range,max_range]]
	        min_range = el 
	        previous_el = el 
	return ranges


run_app(filename)