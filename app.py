import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import dill
import numpy as np
import pandas as pd
import requests
import plotly.figure_factory as ff
import plotly.graph_objs as go

a = dill.load(open('ICD9_flat.pkd','rb'))
Xy = dill.load(open('Xy_CA_2015_loc.pkd','rb'))
op_lib = dill.load(open('op_lib_norm.pkd','rb'))

############### Something to calculate distance ###############
from math import sin, cos, sqrt, atan2, radians

def distcalc(coord, lat, long):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(coord[0])
    lon1 = radians(coord[1])
    lat2 = radians(lat)
    lon2 = radians(long)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # miles conversion
    distance = R * c * 0.621371
    return distance

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
############### End of distance calculation ###############

############### KNN/provider/hcpcs stuff ###############

nn = dill.load(open('nn_cluster.pkd','rb'))
union = dill.load(open('feature_gen.pkd','rb'))
input_gen = dill.load(open('input_gen.pkd','rb'))

def find_match(tag):
    try:
        features = union.transform(input_gen.fit_transform(tag))
        dists, indices = nn.kneighbors(features)
        return sorted([(Xy[x[0]][5],x[1]) for x in zip(indices[0],dists[0])],
                      key=lambda x: x[1])
    except KeyError:
        return [('1235225558', 0)]

    
def most_common(lst):
    return max(set(lst), key=lst.count)

def bin_hcpcs(hcpcs):
    try:
        val = int(hcpcs)
        if val>= 100 and val <= 1999:
            return 'Anesthesia'
        elif val>= 10021 and val <= 69990:
            return 'Surgery'        
        elif val>= 70010 and val <= 79999:
            return 'Radiology'
        elif val>= 80047 and val <= 89398:
            return 'PathoLab'        
        elif val>= 99201 and val <= 99499:
            return 'EvalMgmt'
        else:
            return 'Other'
    except ValueError:
        if hcpcs[-1] == 'T':
            return 'Emerging'
        else:
            return 'Other'

############### End of KNN/provider/hcpcs stuff ###############


############# Loading geocode data from two sets of CSV files (for lat/long and FIPS code) ##############
geocode = pd.read_csv('us-zip-code-latitude-and-longitude.csv',sep=';')
#fipscode = pd.read_csv('ZIP-COUNTY-FIPS_2018-03.csv')
geocode = geocode.set_index('Zip').dropna()
#.join(fipscode.set_index('ZIP')['STCOUNTYFP'])
states_list = [{'label' :x, 'value':x} for x in geocode.State.unique()]
state_to_cities = {x: [{'label':y, 'value': y} for y in geocode[geocode.State == x].City.unique()] for x in geocode.State.unique()}


### lat/lon coordinates retriever from ZIP ###
def get_coord(zipcode):
    try:
        lat = geocode.loc[zipcode].Latitude
        lon = geocode.loc[zipcode].Longitude
        if type(lat)==pd.core.series.Series:
            lat = lat.iloc[0]
            lon = lon.iloc[0]
        coord = [lat, lon]
    except KeyError:
        coord = [34.137557, -118.207650]
    return coord

############## Stuff for making CMS database query
# My SoQL Fetcher, for all my tabulated data from CMS

class SoQLFetcher():
    
    def __init__(self):
        self.ROOT = 'https://data.cms.gov/resource/'
        
        self.ROOT_open = 'https://openpaymentsdata.cms.gov/resource/'

        self.rx_identifier = {
            2016:'xbte-dn4t',
            2015:'x77v-hecv',
            2014:'uggq-gnqc',
            2013:'hffa-2yrd'
        }
        # for years 2016, 2015, 2014 and 2013
    
        self.dx_identifier = {
            2016:'haqy-eqp7',
            2015:'4hzz-sw77',
            2014:'cng4-92f3',
            2013:'5fnr-qp4c',
            2012:'j688-dtru'            
        }
        
        self.payments_identifier = {
            2013: 'tvyk-kca8',
            2014: 'gysc-m9qm',
            2015: 'a482-xr32',
            2016: 'daa6-m7ef'
        }
    
    def retrieve(self,
                year = 2015,
                dataset = 'Rx',
                limit = 25000000,
                select = '*',
                where = None,
                group = None,
                having = None,
                order = None):
        
        if dataset == 'Dx':
            key = '.'.join([self.dx_identifier[year], 'json'])
            addr = ''.join([self.ROOT,key])
        elif dataset == 'Rx':
            key = '.'.join([self.rx_identifier[year], 'json'])
            addr = ''.join([self.ROOT,key])
        else:
            key = '.'.join([self.payments_identifier[year], 'json'])
            addr = ''.join([self.ROOT_open,key])

        if limit:
            limit_query = '='.join(['$limit',str(limit)])
            addr = '?'.join([addr,limit_query])
        
        if select:
            select_query = '='.join(['$select',select])
            addr = '&'.join([addr,select_query])

        if where:
            where_query = '='.join(['$where',where])
            addr = '&'.join([addr,where_query])

        if group:
            group_query = '='.join(['$group',group])
            addr = '&'.join([addr,group_query])
            
        if having:
            having_query = '='.join(['$group',having])
            addr = '&'.join([addr,having_query])
        
        if order:
            order_query = '='.join(['$order',order])
            addr = '&'.join([addr,order_query])
            
        data = requests.get(addr)
        
        return pd.DataFrame(data.json())


def multipleQuery(category, arr):
    
    bracket = '\',\''.join(arr)
    output = ''.join([category,' in ', '(\'', bracket, '\')'])
    return output    
    
############## End of CMS database query helper


############# some code for table

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

############# end code for table


####### some helper code for dataframe apply
def extract_lat(x):
    try:
        data = geocode.loc[x].Latitude
        if type(data) == pd.core.series.Series:
            return data.iloc[0]
        else:
            return data
    except KeyError:
        return 0

def extract_lon(x):
    try:
        data = geocode.loc[x].Longitude
        if type(data) == pd.core.series.Series:
            return data.iloc[0]
        else:
            return data
    except KeyError:
        return 0

def extract_fips(x):
    try:
        data = geocode.loc[x].STCOUNTYFP
        if type(data) == pd.core.series.Series:
            return data.iloc[0]
        else:
            return data
    except KeyError:
        return '00000'
######## end helper code



############### some code for radar chart
def ordered_cats(categories, series):
    data = []
    for elem in categories:
        try:
            data.append(series[elem])
        except KeyError:
            data.append(0)
    return data

def generate_radar(provider_stats,avg_stats):
    
    categories = ['Anesthesia','Surgery', 'Radiology', 'PathoLab', 'EvalMgmt', 'Emerging']
    catlabels = ['Anesthesia','Surgery', 'Radiology', 'Pathology/Lab', 'Evaluation/Management', 'Emerging Technologies']

    data2 = [
        dict(
        type = 'scatterpolar',
        name='My Provider',
        r = ordered_cats(categories,provider_stats),
        theta = catlabels,
        fill = 'toself'
        ),
        dict(
        type = 'scatterpolar',
        name='Specialty avg.',
        r = ordered_cats(categories,avg_stats),
        theta = catlabels,
        fill = 'toself'
        )
    ]

    layout2 = dict(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, int(max(max(provider_stats),max(avg_stats)))+1]
        )
      ),
      showlegend = False
    )

    fig2 = dict(data=data2, layout=layout2)
    fig2['layout'].update(title='Comparison of selected provider to specialty average')
    
    return fig2

############### end code for radar chart

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)


app.config.supress_callback_exceptions=True

app.title = "Find Me A Doc"

app.layout = html.Div([
    html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Link('Dashboard', href='/main'),        
    html.Br(),
    html.A('License For Use', href='https://data.cms.gov/use-agreement?id=sk9b-znav&name=Medicare%20Provider%20Utilization%20and%20Payment%20Data:%20Physician%20and%20Other%20Supplier%20PUF%20CY2015', target="_blank"),
    html.Br(),
    dcc.Link('About the App', href='/overview'),
    html.Br(),
    dcc.Link('About Me', href='/about')
    ],  style = {'columnCount': 7}),    
    html.Div(id='page-content')
    ])
             
main_page = html.Div([
    
    html.H1(children='Find Me A Doc'),
    
    html.Div([
        html.Label('Who can help me with: '),
        dcc.Dropdown(
            options=a,
            value='V72',
            multi=False,
            id='input'
        ),
        html.Div(id='output')
    ]),

    html.Label('I live in: '),
    html.Div([
        html.Label('City: '),
        dcc.Dropdown(
            options=state_to_cities['CA'],
            value='Oakland',
            multi=False,
            id='input_city'
        ),
        html.Label('State: '),
        dcc.Dropdown(
            options=states_list,
            value='CA',
            multi=False,
            id='input_state'
        ),
        html.Label('Zip: '),
        dcc.Dropdown(
            options=[94577],
            value=94577,
            multi=False,
            id='input_zip'
        ),
        html.Label('Preferred Distance: '),
        dcc.Dropdown(
            options=[{'label': '10 miles', 'value': 10},
                    {'label': '25 miles', 'value': 25},
                    {'label': '50 miles', 'value': 50},
                    {'label': '75 miles', 'value': 70},
                    {'label': '100 miles', 'value': 100}],
            value=25,
            multi=False,
            id='input_dist'
        )
    ], style = {'columnCount': 4}),
    
    
    html.Div(id='output_loc'),
    

    #### tab insert here
    
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Matching Providers',
               children=[
                   html.H4(id='prov_name', children='List of providers matching your query'),
                   ## add list of providers here
                   html.Div(id='prov_table')
               ]),

        dcc.Tab(label='Provider Details',
                children=[
                    dcc.Dropdown(
                        options=[{'label':'Placeholder','value':'1235225558','spec':'Internal Medicine'}],
                        value='1235225558',
                        multi=False,
                        id='provider_list'
                    ),

                    html.H4(id='table_name', children='Provider Details'),
                    html.Div(id='table_loc'),
                    
                    dcc.Graph(
                    id='radar_chart'
                    )
                ])

        ])
    
    #### tab end here

    ])


######## Callbacks to handle State/City/Specialization selections

@app.callback(
    Output('input_city','options'),
    [Input('input_state', 'value')]
)
def update_city_list(value):
    return state_to_cities[value]


@app.callback(
    Output('input_city','value'),
    [Input('input_state', 'value')],
    [State('input_city', 'value')]
)
def update_city_list(state, city):
    cities_list = [x['value'] for x in state_to_cities[state]]
    if city in cities_list:
        return city
    else:
        return cities_list[0]

    
@app.callback(
    Output('input_zip','options'),
    [Input('input_city', 'value')],
    [State('input_state', 'value')]
)
def update_zip(city, state):
    selected_city = geocode[geocode.City==city]
    zip_list = list(set(selected_city[selected_city.State==state].index))
    city_to_zip = [{'label':x, 'value': x} for x in zip_list]
    return city_to_zip


@app.callback(
    Output('output', 'children'),
    [Input('provider_list','options')]
)      
def display_output(providers_options):
    providers_specialty = [d['spec'] for d in providers_options]
    return 'Seeking expertise in "{}"'.format(most_common(providers_specialty))


@app.callback(
    Output('output_loc', 'children'),
    [Input('input_city', 'value'),
    Input('input_state', 'value'),
    Input('input_zip', 'value')]
)        
def display_loc(city, state, zipcode):
    return ' '.join([', '.join([city, state]), str(zipcode)])


######## End of State/City/Specialization callbacks


############## List of Procedures Table callback ##################

@app.callback(
    Output('table_loc','children'),
    [Input('provider_list', 'value')]
)
def display_table(provider):
        
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015, select='npi,nppes_credentials,nppes_provider_gender,nppes_provider_first_name,nppes_provider_last_org_name,nppes_provider_street1,provider_type',
                                 where='npi=\'{}\''.format(provider),
                                 limit=1)
    df['Address'] = df['nppes_provider_street1']
    df['Name'] = df.apply(lambda row: row['nppes_provider_first_name'] + ' ' + row['nppes_provider_last_org_name'] + ', ' + row['nppes_credentials'] , axis = 1)
    df['Specialty'] = df['provider_type']
    
    return generate_table(df[['Name','Address','Specialty']])



############## End of List of Procedures Table callback ##################


############## List of Matching Providers Table callback ##################


@app.callback(
    Output('provider_list','options'),
    [Input('input', 'value')]
)
def update_prov_list(icd):
    providers=find_match(icd)
    providers_list = [x[0] for x in providers]
    
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='npi,nppes_provider_first_name as FirstName,nppes_provider_last_org_name as LastName,provider_type as Specialty,nppes_provider_zip',
                                 where=multipleQuery('npi',providers_list),
                                 group='npi,FirstName,LastName,Specialty,nppes_provider_zip'
                                )
    df = df.dropna()
    df['Name'] = df.apply(lambda row: row.FirstName + ' ' + row.LastName, axis = 1)
    
    return [{'label':x[0], 'value': x[1], 'spec': x[2]} for x in zip(df['Name'].values, df['npi'].values, df['Specialty'].values)]

@app.callback(
    Output('prov_table','children'),
    [Input('input', 'value'),
     Input('input_zip', 'value'),
     Input('input_dist', 'value')]
)
def display_providers(icd, zipcode, thresh):
    
    providers=find_match(icd)

    providers_list = [x[0] for x in providers]
    
    score = {x[0]: (2-x[1])/2 for x in providers}
    
    df =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='npi,nppes_provider_first_name as FirstName,nppes_provider_last_org_name as LastName,provider_type as Specialty,nppes_provider_zip',
                                 where=multipleQuery('npi',providers_list),
                                 group='npi,FirstName,LastName,Specialty,nppes_provider_zip'
                                )
    
    
    df = df.dropna()
    df['Name'] = df.apply(lambda row: row.FirstName + ' ' + row.LastName, axis = 1)
    df['Zip'] = df['nppes_provider_zip'].apply(lambda x: int(x[:5]))
    
    coord = get_coord(zipcode)
    df['lat'] = df['Zip'].apply(extract_lat)
    df['long'] = df['Zip'].apply(extract_lon)
    df['dist'] = df.apply(lambda row: gaussian(distcalc(coord, row['lat'],row['long']),0,thresh), axis=1)    
    
    df['Match'] = df['npi'].apply(lambda x: int(100*score[x]))
    
    df['Match'] = df.apply(lambda row: round(row['Match']*row['dist'], 1), axis=1)
    
    df = df[df['Match']>0]
    
    
    return generate_table(df[['Name','Specialty','Zip','Match']].sort_values(by=['Match'], ascending=False).head(10))


############## End of Nearby Providers Table callback ##################

############# Code for radar chart comparing our recommendation to specialty-based search #############
@app.callback(
    Output('radar_chart','figure'),
    [Input('provider_list', 'value')],
    [State('provider_list', 'options'),
    State('input_state', 'value')]
)
def compare_provider(your_provider, matching_provider, state):
    specialty = most_common([d['spec'] for d in matching_provider])
    
    df_provider =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='npi,hcpcs_code',
                                 where='npi=\"{}\"'.format(your_provider),
                                 group='npi,hcpcs_code'
                                )
    
    df_provider['hcpcs_code'] = df_provider['hcpcs_code'].apply(bin_hcpcs)
    provider_stats = df_provider.groupby(['npi', 'hcpcs_code']).hcpcs_code.count().mean(level='hcpcs_code')
    
    df_specialty =  SoQLFetcher().retrieve(dataset = 'Dx', year=2015,
                                 select='npi,hcpcs_code',
                                 where='provider_type=\"{}\" AND nppes_provider_state=\"{}\"'.format(specialty,state),
                                 group='npi,hcpcs_code'
                                )
    
    
    df_specialty = df_specialty.dropna()
    df_specialty['hcpcs_code'] = df_specialty['hcpcs_code'].apply(bin_hcpcs)
    specialty_profile = df_specialty.groupby(['npi', 'hcpcs_code']).hcpcs_code.count()
    avg_stats = specialty_profile.mean(level='hcpcs_code')
    
    return generate_radar(provider_stats,avg_stats)

############# End of radar chart code #############


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/about':
         return html.Div([
             html.H1(children='About the author'),
             html.P(children='E-mail: dong.jin.shin@mg.thedataincubator.com'),
             html.Img(id='image2', src='/static/profile_pic.jpg', height= 300),
             html.P(children='Dong Jin (DJ) hails from New Zealand, though he has spent the better part of the last decade in Baltimore, MD. He received his bachelors in Biomedical Engineering and Electrical and Computer Engineering from Duke University. He went on to receive his doctorate in Biomedical Engineering from The Johns Hopkins University in 2016. While at Hopkins, he led the development of a portable diagnostic platform for pathogen detection in a hospital emergency department. He subsequently took on various assignments during his postdoc years to drive commercialization of nascent technologies from his lab. He joined the Data Incubator in September 2018 with the goal of transitioning his career into the private sector by acquiring skills that will help drive business outcomes at scale.')
         ], style={'marginBottom': 25, 'marginTop': 25})
        
    elif pathname == '/overview':
         return html.Div([
             html.H1(children='Workflow Overview'),
             html.Img(id='image1', src='/static/workflow_1.jpg')
         ])
    else:
        return main_page


if __name__ == '__main__':
    app.run_server()