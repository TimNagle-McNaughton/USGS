#!/usr/bin/env python
# coding: utf-8

# **Use the suppplied requirements.txt to set up your virtual env.**
# 
# Using pip:
# 
# >pip install -r requirements.txt
# 
# Using Conda:
# 
# >conda create --name (env_name) --file requirements.txt

# In this notebook we will explore how you can access, query, and use the Syngine models available via IRIS. These models can generate synthetic seismograms for any event/observation pair on Earth.

# In[1]:


# import obspy
import obspy
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime, Stream
from obspy.geodetics import locations2degrees

# import the two different clients, and differentiate between names
from obspy.clients.fdsn import Client as Client 
from obspy.clients.syngine import Client as synClient

# other packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
import pandas as pd
import requests
import ssl 

# matplotlib magic
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# having some trouble with my work computer's SSL certs so let's hack around that
try:
    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:
    pass

else:
    ssl._create_default_https_context = _create_unverified_https_context


# In[3]:


# set an observatory to use for later analysis
network = "IU"
station = "ANMO"

# get a real station.
irisClient = Client("IRIS")
ANMO = irisClient.get_stations(network=network, 
                           station=station, 
                           format="text")[0][0]

print(ANMO)


# We can load all the current models that are available on Syngine, and preview some basic information about them.

# In[4]:


# initialize the syngine client
client = synClient()

# load the models
models = client.get_available_models()


# Here are the models we'll be using:
# 
# | Model name | Resolution (s) | Description |
# |:---:|:---:|:---:|
# | ak135f_1s | 1-100 | AK135 with density & Q of [Montagner & Kennet (1996)](https://academic.oup.com/gji/article/125/1/229/703026?login=true) |
# | ak135f_2s | 2-100 | AK135 with density & Q of [Montagner & Kennet (1996)](https://academic.oup.com/gji/article/125/1/229/703026?login=true) |
# | ak135f_5s | 5-100 | AK135 with density & Q of [Montagner & Kennet (1996)](https://academic.oup.com/gji/article/125/1/229/703026?login=true) | 
# | iasp91_2s | 2-100 | IASP91  |
# | prem_a_2s | 2-100 |  anisotropic PREM  | 
# | prem_a_10s | 10-100 |  anisotropic PREM  | 
# | prem_a_20s | 20-100 |  anisotropic PREM  |
# | prem_a_5s | 5-100 |  anisotropic PREM  |
# | prem_i_2s | 2-100 | isotropic PREM |

# Full details are available [online](http://ds.iris.edu/ds/products/syngine/). This same data is actually stored in the models dictionary we just generated. (Click the output to toggle scrolling on the output in notebooks)

# In[5]:


# print the models. Not using print() will maintain the dict structure, so we will just invoke the variable itself.
models


# These models are slightly different in their interpretation of the Earth's interior. We can quickly hack together a function and a plot to show these differences. The models are available online as text files that we'll need to parse for the correct data.

# In[6]:


# define a function to parse the giant model text files
def get_model(url):
    
    # read the text data from the URLs
    data = requests.get(url).text
    
    # make an empty list to store results
    d = []
    
    # loop through each line
    for line in data.splitlines():
        
        # do some cleaning
        line = line.strip()
        
        # ignore blanks or comments
        if not line or line.startswith("#"):
            continue
        
        # split lines further
        line = line.split()
        
        # skip short lines
        if len(line) != 6:
            continue
        
        # read in numbers as floats
        line = [float(_i) for _i in line]
        
        # save a dictionary of model parameters to the data list
        d.append({"depth": (6371000. - line[0]) / 1000.0,
                  "v_p": line[2] / 1000.0,
                  "v_s": line[3] / 1000.0,
                  "rho": line[2] / 1000.0})
    
    # return the pandas dataframe
    return pd.DataFrame(d)


# We can now scrape the three parent models:

# In[7]:


# download and scrape the models' data
ak135f = get_model("http://ds.iris.edu/media/product/emc-syngine/files/1dmodel_ak135f.txt")
prem = get_model("http://ds.iris.edu/media/product/emc-syngine/files/1dmodel_PREMiso.txt")
iasp91 = get_model("http://ds.iris.edu/media/product/emc-syngine/files/1dmodel_iasp91.txt")


# Let's plot the models' data using some lovely matplotlib. This basically recreates the velocity/depth profiles from Montagner & Kennet (1996).

# In[8]:


# write a function to plot profiles
def plot_profile(index, depth_min, depth_max, v_min, v_max):
 
    # plot the velocities and depth for each model
    ax[index].plot(ak135f.v_p, ak135f.depth, color="0.0", ls="-", lw=2, label="ak135-F")
    ax[index].plot(ak135f.v_s, ak135f.depth, color="0.0", ls="-", lw=2)
    ax[index].plot(iasp91.v_p, iasp91.depth, color="0.4", ls="--", lw=2, label="iasp91")
    ax[index].plot(iasp91.v_s, iasp91.depth, color="0.4", ls="--", lw=2)
    ax[index].plot(prem.v_p, prem.depth, color="0.8", ls=":", lw=2, label="PREM")
    ax[index].plot(prem.v_s, prem.depth, color="0.8", ls=":", lw=2)

    # set the labels
    ax[index].set_ylabel("Depth (km)")
    ax[index].set_xlabel("Velocity (km/s)")
    ax[index].set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax[index].set_ylim(depth_min, depth_max)
    ax[index].set_xlim(v_min, v_max)

    # switch the ticks and labels to the left side
    ax[index].yaxis.tick_right()
    ax[index].yaxis.set_label_position("right")
    
    # make the y axes descending
    ax[index].invert_yaxis()


# In[9]:


# set up the plot
fig = plt.figure(figsize=(20, 8))

# set up the subplot layout
ax = fig.subplots(nrows=1, ncols=4)
    
# plot the profiles
plot_profile(0, 5, 6000, 0, 14)
plot_profile(1, 2, 50, 2, 9)
plot_profile(2, 2885, 2895, 0, 12)
plot_profile(3, 5140, 5170, 0, 12)

# add the legend
ax[3].legend(loc="upper left", fancybox=True, frameon=True)

# show the composite plot
plt.show();


# We can now select the event we want to examine. We'll use [the largest earthquake so far in 2021](https://earthquake.usgs.gov/earthquakes/eventpage/us7000dflf/executive) ([alternate link](https://ds.iris.edu/spud/eventplot/18822380)): 
# 
# 2021-03-04 19:28:33 M8.1 Kermadec Islands, New Zealand
# 
# - 29.723°S, 177.279°W
# 
# - 28.9 km depth

# In[10]:


# load basic event information
event_dict = {"lat": -29.723,
              "lon": 177.279,
              "depth": 28.9}

# load the event IDs for later
event_id_iris = "18822380"
event_id = "GCMT:C202103041928A"

# set the start and endtime, add a significant delay
starttime = UTCDateTime("2021-03-04T19:41:33.000000Z")
endtime = UTCDateTime("2021-03-04T19:56:00.200000Z")


# For some reason, obspy will not take this event's ID, not sure what the root cause of the error is.

# In[11]:


# try getting the event by its ID
try:
    
    # use the event ID from the IRIS page: https://ds.iris.edu/spud/eventplot/18822380
    event = irisClient.get_events(eventid=event_id_iris, catalog="NEIC PDE")

# print the error
except Exception as error:
    print("Error:", error)


# With the correct imports ([Cartopy](https://scitools.org.uk/cartopy/docs/latest/)), you can rapidly generate overview plots. Let's examine all events on March 4th during the 15-minute window set above. Note that the obspy uses body magnitude (mb) for both querying and plotting.

# In[12]:


# get all the events between our start and end times 
events = irisClient.get_events(starttime=starttime, endtime=endtime)

# plot. Note the semicolon--it prevents auto-outputs with matplotlib in notebooks
events.plot(projection="global");


# Since we can't use the event ID, we can just set some very narrow query parameters to get our event.

# In[13]:


# use narrow time ranges to get the event we want
event = irisClient.get_events(starttime=starttime, endtime=endtime, minmagnitude=5.5)

# plot. Note the semicolon
event.plot(projection="ortho");


# Load the [TauPyModel](https://docs.obspy.org/packages/obspy.taup.html) which was dervied from [Crotwell *et al*, 1999](https://pubs.geoscienceworld.org/ssa/srl/article-abstract/70/2/154/142385/The-TauP-Toolkit-Flexible-Seismic-Travel-time-and?redirectedFrom=fulltext).
# 
# The package ships with 13 possible models (more than syngine has currently). Since we have three ak135 models in syngine we'll use that for visualization. You can also just call "prem" here, but there's virtually no difference for visual purposes.

# In[14]:


# set the model to use and intialize it
tauModel = "ak135"
model = TauPyModel(tauModel)


# Now that we have a model, we can [estimate the travel times](https://docs.obspy.org/sphinx3/packages/autogen/obspy.taup.tau.TauPyModel.get_travel_times_geo.html) for each arriving ray from our event to our station. This returns a chronological list of arrivals.

# In[15]:


# get the travel times, use the station object and our event dictionary
travel_time = model.get_travel_times_geo(source_depth_in_km=event_dict["depth"], 
                                         source_latitude_in_deg=event_dict["lat"],
                                         source_longitude_in_deg=event_dict["lon"], 
                                         receiver_latitude_in_deg=ANMO.latitude,
                                         receiver_longitude_in_deg=ANMO.longitude)

# print the travel times
print(travel_time)


# The [obspy.taup.tau.Arrivals class](https://docs.obspy.org/packages/autogen/obspy.taup.tau.Arrivals.html) is hard to work with, or maybe I'm missing something obvious. Here we'll just scrape out the phases so we can visualize them.

# In[16]:


# create an empty list to store phases
phase_list = []

# loop through the arrivals
for time in travel_time:
    
    # split the output by spaces
    split = str(time).split(" ")
    
    # get the first item, the phase
    phase = split[0]
    
    # add to to the phase list
    phase_list.append(phase)


# We can also visualize the paths of these rays using an [another function](https://docs.obspy.org/packages/autogen/obspy.taup.tau.TauPyModel.get_ray_paths_geo.html). You can pass "plot_all" to this fucntion to see all the phases without the cell above, but now that we have a list of all the phases we can index and slice them if we want to.

# In[17]:


# calculate the ray paths for 5 arrivals, use the same calls as above
ray_paths = model.get_ray_paths_geo(source_depth_in_km=event_dict["depth"], 
                                    source_latitude_in_deg=event_dict["lat"],
                                    source_longitude_in_deg=event_dict["lon"], 
                                    receiver_latitude_in_deg=ANMO.latitude,
                                    receiver_longitude_in_deg=ANMO.longitude,
                                    phase_list=phase_list)


# In[18]:


# visualize the ray paths. Again note the semicolon.
ray_paths.plot_rays();


# In[19]:


# calculate the distance between the event and detection
distance = locations2degrees(lat1=event_dict["lat"], long1=event_dict["lon"], 
                             lat2=ANMO.latitude, long2=ANMO.longitude)

# print the results
print("{} degrees.".format(round(distance, 2)))


# We can now download waveform data for the event from each model using the syngine client function [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.syngine.client.Client.get_waveforms.html). This function is essentially the same as [obspy.clients.fdsn.client.Client.get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html) so you will likely recognize the format.

# In[20]:


# select the component we want to model
component = "Z"

# set the sampling rate, only upsampling is allowed so must be equal to or larger than the instruments'
sample_rate = 0.05

# create an empty dictionary to store the results from each model
data = {}

# loop through the 9 available models
for model in models.keys():
    
    # Add a printing call since these can take some time
    print("Processing {}...".format(model))
    
    # some of the models can fail for an unknown reason (related to the time interval?)
    try:
    
        # download the data for each model into the dictionary
        data[model] = client.get_waveforms(model=model, network=network, 
                                           station=station, components=component, 
                                           dt=sample_rate, eventid=event_id,
                                           starttime=starttime, endtime=endtime)
    # print the error if we hit one
    except Exception as error:
        print("Error in {}:".format(model),  error)


# We can compare the model results to the real observation, so we need to collected the actual waveform from ANMO for this event.

# In[21]:


# initialize a stream
stream = Stream()

# we can use some of the same variables as above
stream = irisClient.get_waveforms(network, station, "00", "BHZ", starttime, endtime)


# Let's make sure the stream looks like it's correct.

# In[22]:


# plot the stream
stream.plot();


# That looks pretty good, but it's noisy, we can apply some basic filtering ([Butterworth-bandpass](https://docs.obspy.org/packages/autogen/obspy.signal.filter.bandpass.html)) to improve the results. Since we want to avoid filter ringing we'll modify the get_waveforms call above and trim the results.

# In[23]:


# initialize another stream
stream_proc = Stream()

# overwrite the above stream, and request additional data at the start
stream_proc = irisClient.get_waveforms(network, station, "00", "BHZ", starttime-20, endtime)

# filter the stream with a basic bandpass filter
stream_proc.filter("bandpass", freqmin=0.05, freqmax=0.1)

# trim the extra off the beginning
stream_proc.trim(starttime, endtime)

# plot the filtered results
stream_proc.plot();


# Note the change in the y-axis between the two plots, we'll have to account for these when we plot the streams together. Let's take a look at one of the synthetic waveforms.

# In[24]:


synth_example = data["ak135f_1s"]

# filter the stream with a basic bandpass filter
synth_example.filter("bandpass", freqmin=0.05, freqmax=0.1)

synth_example.plot();


# In[25]:


# add the actual observations to our data dictionary
data['Observed'] = stream


# Time to do some plotting. Prepare yourself for a lot of matplotlib fun.

# In[26]:


# set up the figure
fig = plt.figure(figsize=(16, 6), tight_layout=True)

# add our two subplots, this is the least gross way I know of
ax = fig.subplots(nrows=1, ncols=2)

# set a scale factor for the model traces
factor = 1E3

# set a factor to scale down the observed trace
actual_factor = 1.5E-5

# loop through our models
for index, model in enumerate(sorted(data.keys())):
    
    # get the trace from the stored downloaded model data
    tr = data[model][0]
    
    # apply the same bandpass filter to all waveforms
    tr = tr.filter("bandpass", freqmin=0.05, freqmax=0.1) # feel like this is probably wrong...
    
    # get the vertical position for the trace to be plotted (top to bottom)
    pos = len(data) - index - 1
    
    # use different scaling and colors for the actual data
    if model == "Observed":
        
        # plot the actual data
        ax[0].plot(tr.times(), tr.data * actual_factor + pos, color="r", lw=1)
        ax[1].plot(tr.times(), tr.data * actual_factor + pos, color="r", lw=1)

    # plot the models
    else:
        
        # plot the traces for each model
        ax[0].plot(tr.times(), tr.data * factor + pos, color="k", lw=1)
        ax[1].plot(tr.times(), tr.data * factor + pos, color="0.1", lw=1)
    
    # we need one line for each model
    ax[0].set_yticks(list(range(len(data))))
    ax[1].set_yticks(list(range(len(data))))
    
    # add grid
    ax[0].xaxis.grid(True)
    ax[1].xaxis.grid(True)
    
    # set the labels to be blank here
    ax[0].set_yticklabels([""] * len(data))
    ax[1].set_yticklabels([""] * len(data))
    
    # set labels
    ax[0].set_xlabel("Time since event origin (s)", fontsize=14)
    ax[0].set_title("Body waves and inital surface waves", fontsize=18)
    ax[1].set_xlabel("Time since event origin (s)", fontsize=14)
    ax[1].set_title("First body waves?...", fontsize=18)

    # zoom in on the first body waves
    ax[1].set_xlim(0, 300)
    
    # label the models
    ax[0].text(0, pos, model, color="white",
             bbox=dict(facecolor="black", edgecolor="None"),
             ha="right", fontsize=14)

# save and display the figure
plt.savefig("syngine_compare_models.png", dpi=300, facecolor="white")
plt.show()

